/**
 * \file PVNrawDiskBackend.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef PVNRAWDISKBACKEND_H_
#define PVNRAWDISKBACKEND_H_

#include <fcntl.h>
#include <sys/stat.h>

#include <sstream>
#include <string>

#include <tbb/tick_count.h>

#include <pvkernel/core/PVAllocators.h>
#include <pvkernel/core/PVMatrix.h>

const std::string INDEX_FILENAME = std::string("nraw.idx");

namespace PVRush {

/*
 *
 * File policy classes
 *
 *
 */

struct RawFilePolicy
{
	typedef int file_t;

	bool Open(std::string const& filename, file_t* file, bool direct = true)
	{
		int64_t flags = O_RDWR | O_CREAT;
		if (direct) {
			flags |= O_DIRECT;
		}
		*file = open(filename.c_str(), flags, 0640);
		return *file != -1;
	}

	inline int64_t Write(const char* content, uint64_t buf_size, file_t file)
	{
		return write(file, content, buf_size);
	}

	inline int64_t Read(file_t file, void* buffer,  uint64_t buf_size)
	{
		tbb::tick_count t1 = tbb::tick_count::now();
		int64_t r = read(file, buffer, buf_size);
		tbb::tick_count t2 = tbb::tick_count::now();
		_read_interval += (t2-t1);

		return r;
	}

	inline int64_t ReadAt(file_t file, uint64_t offset, void* buffer, uint64_t buf_size)
	{
		lseek(file, offset, SEEK_SET);
		return Read(file, buffer, buf_size);
	}

	inline int64_t Seek(file_t file, int64_t offset)
	{
		return lseek(file, offset, SEEK_CUR);
	}

	inline int64_t Tell(file_t file)
	{
		return lseek(file, 0, SEEK_CUR);
	}

	inline uint64_t Size(file_t file)
	{
		int64_t pos = Tell(file);
		uint64_t size = lseek(file, 0, SEEK_END);
		lseek(file, pos, SEEK_SET);
		return size;
	}

	void Flush(file_t)
	{
	}

	void Close(file_t file)
	{
		close(file);
	}

	tbb::tick_count::interval_t _read_interval;
};

struct BufferedFilePolicy
{
	typedef FILE* file_t;

	bool Open(std::string const& filename, file_t* file)
	{
		*file = fopen(filename.c_str(), "rw");
		return *file != nullptr;
	}

	inline int64_t Write(const char* content, uint64_t buf_size, file_t file)
	{
		return fwrite(content, buf_size, 1, file);
	}

	inline int64_t Read(file_t file, void* buffer, uint64_t buf_size)
	{
		tbb::tick_count t1 = tbb::tick_count::now();
		int64_t r = fread(buffer, 1, buf_size, file);
		tbb::tick_count t2 = tbb::tick_count::now();
		_read_interval += (t2-t1);
		return r;
	}

	inline int64_t ReadAt(file_t file, uint64_t offset, void* buffer, uint64_t buf_size)
	{
		fseek(file, offset, SEEK_SET);
		return Read(file, buffer, buf_size);
	}

	inline int64_t Seek(file_t file, int64_t offset)
	{
		return fseek(file, offset, SEEK_CUR);
	}

	inline int64_t Tell(file_t file)
	{
		return ftell(file);
	}

	void Flush(file_t file)
	{
		fflush(file);
	}

	void Close(file_t file)
	{
		fclose(file);
	}

	tbb::tick_count::interval_t _read_interval;
};

/*
 *
 * Field policy classes
 *
 *
 */

template <typename FilePolicy = RawFilePolicy>
class PVNRawDiskBackend : public FilePolicy
{
	static constexpr uint64_t BUF_ALIGN = 512;
	static constexpr uint64_t READ_BUFFER_SIZE = 512*1024 + BUF_ALIGN;
	static constexpr uint64_t NB_CACHE_BUFFERS = 3;
	static constexpr uint64_t INVALID = UINT64_MAX;

private:
	struct offset_fields_t
	{
		uint64_t offset = 0;
		uint64_t field = 0;
	};

public:
	typedef typename FilePolicy::file_t file_t;
	//typedef std::pair<uint64_t, uint64_t> offset_fields_t;
	typedef PVCore::PVMatrix<offset_fields_t, PVRow, PVCol> index_table_t;
	typedef PVNRawDiskBackend<FilePolicy> this_type;

public:
	PVNRawDiskBackend(const char* nraw_folder, uint64_t num_cols) :
		_nraw_folder(nraw_folder),
		_num_cols(num_cols),
		_cache_pool(*this)
	{
		_columns.reserve(_num_cols);
		for (int col = 0 ; col < _num_cols ; col++) {
			_columns.emplace_back();
			PVColumn& column = _columns.back();

			// Open file
			std::stringstream filename;
			filename << nraw_folder << "/column_" << col;
			column.filename = filename.str();
			if(!this->Open(column.filename, &column.file)) {
				PVLOG_ERROR("PVNRawDiskBackend: Error opening file %s (%s)\n", filename.str().c_str(), strerror(errno));
				return;
			}

			// Create buffer
			uint64_t buffer_size = _write_buffers_size_pattern[0];
			column.buffer_write = PVCore::PVAlignedAllocator<char, BUF_ALIGN>().allocate(buffer_size);
			column.buffer_write_ptr = column.buffer_write;
			column.buffer_write_end_ptr = column.buffer_write + buffer_size;
			column.field_length = 0; // Or any value grater than 0 to specify a fixed field length;
		}

		_indexes.resize(_next_indexes_nrows, num_cols);
		_next_indexes_nrows += _index_fields_size_pattern[++_fields_size_idx];
	}

	uint64_t add(PVCol col_idx, const char* field, uint64_t field_size)
	{
		PVColumn& column = get_col(col_idx);
		const uint64_t written_field_size = field_size + column.end_char();
		uint64_t field_part2_size = 0;
		char* field_part2 = nullptr;
		uint64_t write_size = 0;

		// Index field
		if (column.fields_ignored_size + written_field_size > READ_BUFFER_SIZE) {
			uint64_t field_offset_in_file = this->Tell(column.file) + (column.buffer_write_ptr - column.buffer_write) ;

			// Resize indexes matrix if needed
			if (column.fields_indexed == _indexes.get_nrows()) {
				tbb::tick_count t1 = tbb::tick_count::now();
				_indexes.resize_nrows(_next_indexes_nrows);
				tbb::tick_count t2 = tbb::tick_count::now();
				_matrix_resize_interval += (t2-t1);
				uint64_t index = std::min(++_fields_size_idx, _max_fields_size_idx);
				_next_indexes_nrows += _index_fields_size_pattern[index];
			}

			offset_fields_t offset_field;
			offset_field.offset = field_offset_in_file;
			offset_field.field = column.fields_nb;
			_indexes.set_value(column.fields_indexed, col_idx, offset_field);
			column.fields_ignored_size = 0;
			column.fields_indexed++;
		}
		column.fields_ignored_size += written_field_size;
		_indexes_nrows = std::max(column.fields_indexed, _indexes_nrows);
		column.fields_nb++;

		// Fill the buffer with complete field
		if (column.buffer_write_ptr + written_field_size <= column.buffer_write_end_ptr) {
			memcpy(column.buffer_write_ptr, field, field_size);
			column.buffer_write_ptr[field_size] = 0;
			column.buffer_write_ptr += field_size+1;
		}
		// Fill the buffer_write with splitted field
		else {
			uint64_t field_part1_size = column.buffer_write_end_ptr - column.buffer_write_ptr;
			memcpy(column.buffer_write_ptr, field, field_part1_size);
			field_part2 = (char *)(field + field_part1_size);
			field_part2_size = field_size - field_part1_size;
			column.buffer_write_ptr += field_part1_size;
		}

		// Write buffer_write to disk
		if (column.buffer_write_ptr == column.buffer_write_end_ptr) {
			write_size = this->Write(column.buffer_write, _write_buffers_size_pattern[column.buffers_write_size_idx], column.file);
			if(write_size <= 0) {
				PVLOG_ERROR("PVNRawDiskBackend: Error writing column %d to disk (%s)\n", col_idx, strerror(errno));
				return 0;
			}

			uint64_t buffer_max_size = _write_buffers_size_pattern[column.buffers_write_size_idx];

			// Reallocate a bigger buffer_write and copy the end of splitted field
			if (column.buffers_write_size_idx < _max_write_size_idx) {
				uint64_t new_buffer_max_size = _write_buffers_size_pattern[++column.buffers_write_size_idx];
				PVCore::PVAlignedAllocator<char, BUF_ALIGN>().deallocate(column.buffer_write, buffer_max_size);
				column.buffer_write = PVCore::PVAlignedAllocator<char, BUF_ALIGN>().allocate(new_buffer_max_size);
				column.buffer_write_end_ptr = column.buffer_write + new_buffer_max_size;

				memcpy(column.buffer_write, field_part2, field_part2_size);
				column.buffer_write[field_part2_size] = 0;
				column.buffer_write_ptr = column.buffer_write + field_part2_size + 1;
			}
			// Recycle previously allocated buffer_write
			else {
				column.buffer_write_ptr = column.buffer_write;
			}
		}

		return write_size;
	}

	void flush()
	{
		// direct mode prevents to write a buffer of an arbitrary size
		set_direct_mode(false);

		for (uint64_t col_idx = 0 ; col_idx < _num_cols; col_idx++) {
		 	PVColumn& column = get_col(col_idx);
			uint64_t partial_buffer_size = column.buffer_write_ptr - column.buffer_write;
			if (partial_buffer_size > 0) {
				if(!this->Write(column.buffer_write, partial_buffer_size, column.file)) {
					PVLOG_ERROR("PVNRawDiskBackend: Error writing column %d to disk (%s)\n", col_idx, strerror(errno));
					return;
				}
			}
			PVCore::PVAlignedAllocator<char, BUF_ALIGN>().deallocate(column.buffer_write, _write_buffers_size_pattern[column.buffers_write_size_idx]);
			column.reset();
		}

		store_index_to_disk();
	}

	const char* at(PVRow field, PVCol col, size_t& size_ret)
	{
		PVColumn& column = get_col(col);

		uint64_t nb_fields_left = _cache_pool.get_cache(field, col);

		return next(col, nb_fields_left, column.buffer_read, size_ret);
	}

	inline const char* next(uint64_t col, size_t& size_ret)
	{
		PVColumn& column = get_col(col);
		return next(col, 1, column.buffer_read_ptr, size_ret);
	}

	uint64_t search_in_column(uint64_t col_idx, std::string const& field)
	{
		PVColumn& column = get_col(col_idx);

		uint64_t chunk_size = _write_buffers_size_pattern[_max_write_size_idx];
		uint64_t total_read_size = 0;
		uint64_t nb_occur = 0;
		uint64_t read_size = 0;

		char* const buffer = PVCore::PVAlignedAllocator<char, BUF_ALIGN>().allocate(chunk_size);

		char* buffer_ptr = buffer + BUF_ALIGN;
		bool last_chunk = false;
		do
		{
			read_size = this->Read(column.file, buffer+BUF_ALIGN, chunk_size-BUF_ALIGN);
			last_chunk = read_size < (chunk_size-BUF_ALIGN);

			// TODO: clean this mess
			while (true) {
				char* endl = nullptr;
				if (last_chunk) {
					endl = (char*) memchr(buffer_ptr, '\0', chunk_size);
					if (endl == nullptr || buffer_ptr >= buffer+BUF_ALIGN+read_size) {
						buffer_ptr = buffer;
						break;
					}
				}
				else {
					endl = (char*) memchr(buffer_ptr, '\0', chunk_size);
					if (endl == nullptr) {
						uint64_t partial_line_length = buffer+chunk_size-buffer_ptr;
						char* dst = buffer+BUF_ALIGN-partial_line_length;
						memcpy(dst, buffer_ptr, partial_line_length);
						buffer_ptr = dst;
						break;
					}
				}

				int64_t line_length = (&endl[0] - &buffer_ptr[0]);
				nb_occur += (memcmp(buffer_ptr, field.c_str(), field.length()) == 0);
				buffer_ptr += (line_length+1);
			}
			total_read_size += read_size;

		} while(!last_chunk);

		PVCore::PVAlignedAllocator<char, BUF_ALIGN>().deallocate(buffer, chunk_size);

		return nb_occur;
	}

	void set_direct_mode(bool direct = true)
	{
		for (int col = 0 ; col < _num_cols ; col++) {
			PVColumn& column = get_col(col);
			uint64_t pos = this->Tell(column.file);
			this->Close(column.file);
			this->Open(column.filename, &column.file, direct);
			this->Seek(column.file, pos);
		}
		_direct_mode = direct;
	}

	void store_index_to_disk()
	{
		char* data = (char*) _indexes.get_data();
		file_t file;
		std::stringstream filename;
		filename << _nraw_folder << "/" << INDEX_FILENAME;
		this->Open(filename.str(), &file, false);
		uint64_t size = _num_cols * _indexes_nrows * sizeof(offset_fields_t);
		if (size > 0) {
			int64_t write_size = this->Write(data, size, file);
			if(write_size <= 0) {
				PVLOG_ERROR("PVNRawDiskBackend: Error writing index to disk [size=%d] (%s)\n", size, strerror(errno));
				return;
			}
		}
		this->Close(file);
	}

	void load_index_from_disk()
	{
		set_direct_mode(false);

		std::stringstream filename;
		filename << _nraw_folder << "/" << INDEX_FILENAME;
		file_t file;
		this->Open(filename.str(), &file, false);
		uint64_t size = this->Size(file);
		_indexes_nrows = size / _num_cols / sizeof(offset_fields_t);
		std::cout << "size=" << size << std::endl;
		_indexes.resize(_indexes_nrows, _num_cols);
		char* data = (char*) _indexes.get_data();
		int64_t read_size = this->Read(file, data, size);
		std::cout << "read_size=" << read_size << std::endl;
		if(read_size <= 0) {
			PVLOG_ERROR("PVNRawDiskBackend: Error reading index from disk (%s)\n", strerror(errno));
			return;
		}
		this->Close(file);
	}

	void print_stats()
	{
		PVLOG_INFO("read: %d msec\n", this->_read_interval.seconds()*1000);
		PVLOG_INFO("search: %d msec\n", this->_search_interval.seconds()*1000);
		PVLOG_INFO("resize: %d msec\n", this->_matrix_resize_interval.seconds()*1000);
	}

	void print_indexes()
	{
		for (uint64_t r=0; r < _indexes_nrows; r++) {
			for (uint64_t c=0; c < _num_cols; c++) {
				PVLOG_INFO("index[%d][%d] = %d (offset), %d (field)\n", r, c, _indexes.at(r, c).offset, _indexes.at(r, c).field);
			}
		}
	}

	~PVNRawDiskBackend()
	{
		// Close files
		for (uint64_t col_idx = 0 ; col_idx < _num_cols; col_idx++) {
			PVColumn& column = get_col(col_idx);
			this->Close(column.file);
		}
	}

private:
	inline char* next(uint64_t col, uint64_t nb_fields, char* buffer, size_t& size_ret)
	{
		tbb::tick_count t1 = tbb::tick_count::now();

		PVColumn& column = get_col(col);

		if (buffer == nullptr) return nullptr;

		// Extract proper field from buffer
		char* buffer_ptr = buffer;
		if (column.field_length > 0) {
			buffer_ptr += nb_fields * column.field_length;
			size_ret = column.field_length;
		}
		else {
			char* end_field_ptr = nullptr;
			uint64_t size_to_read = READ_BUFFER_SIZE - (buffer - column.buffer_read_ptr);
			for (uint64_t i = 0; i < nb_fields; i++) {
				end_field_ptr = (char*) memchr(buffer_ptr, '\0', size_to_read);
				assert(end_field_ptr && (end_field_ptr - column.buffer_read < READ_BUFFER_SIZE));
				uint64_t field_length = (end_field_ptr - buffer_ptr)+1;
				size_to_read -= field_length;
				buffer_ptr += field_length;
			}
			size_ret = strnlen(buffer_ptr, size_to_read);
		}

		column.buffer_read_ptr = buffer_ptr;

		tbb::tick_count t2 = tbb::tick_count::now();
		_search_interval += (t2-t1);
		return buffer_ptr;
	}


private:
	struct PVColumn
	{
	public:
		PVColumn() { reset(); }
		bool end_char() { return field_length == 0; }

		void reset()
		{
			buffer_write = nullptr;
			buffer_write_ptr = nullptr;
			buffer_write_end_ptr = nullptr;
			buffers_write_size_idx = 0;

			field_length = 0;
			fields_nb = 0;
			fields_ignored_size = 0;
			fields_indexed = 0;

			buffer_read = nullptr;
			buffer_read_ptr = nullptr;
		}
	public:
		file_t file = 0;
		std::string filename;

		char* buffer_write;
		char* buffer_write_ptr;
		char* buffer_write_end_ptr;
		uint64_t buffers_write_size_idx;

		uint64_t field_length;
		uint64_t fields_nb;
		uint64_t fields_ignored_size;
		uint64_t fields_indexed;

		char* buffer_read;
		char* buffer_read_ptr;
	};

	template <typename T>
	class PVCachePool
	{
	public:

		PVCachePool(PVRush::PVNRawDiskBackend<T>& parent) : _parent(parent)
		{
			for (uint64_t cache_idx = 0; cache_idx < NB_CACHE_BUFFERS; cache_idx++) {
				_caches[cache_idx].buffer = PVCore::PVAlignedAllocator<char, BUF_ALIGN>().allocate(READ_BUFFER_SIZE);
			}
		}

		uint64_t get_cache(uint64_t field, uint64_t col)
		{
			bool cache_miss = true;
			uint64_t cache_idx = 0;

			// Is there a cache for this column?
			for (; cache_idx < NB_CACHE_BUFFERS && _caches[cache_idx].column != col; cache_idx++) {}

			// Yes
			if (cache_idx < NB_CACHE_BUFFERS) {
				PVReadCache& cache = _caches[cache_idx];
				cache_miss = (field < cache.first_field || field > cache.last_field);
			}
			// No: find LRU cache
			else {
				tbb::tick_count older_timestamp = tbb::tick_count::now();
				for (uint64_t i = 0; i < NB_CACHE_BUFFERS; i++) {
					PVReadCache& cache = _caches[i];
					tbb::tick_count::interval_t interval = cache.timestamp - older_timestamp;
					if (interval.seconds() < 0) {
						older_timestamp = cache.timestamp;
						cache_idx = i;
					}
				}
				 _caches[cache_idx].column = col;
			}

			PVReadCache& cache = _caches[cache_idx];
			PVColumn& column = _parent.get_col(col);

			// Fetch data from disk
			char* buffer_ptr = cache.buffer;
			if (cache_miss) {
				int64_t field_index = get_index(col, field);
				uint64_t disk_offset = field_index == -1 ? 0 : _parent._indexes.at(field_index, col).offset;
				uint64_t aligned_disk_offset = disk_offset;
				if (_parent._direct_mode) {
					aligned_disk_offset = (disk_offset / BUF_ALIGN) * BUF_ALIGN;
					buffer_ptr += (disk_offset - aligned_disk_offset);
				}
				int64_t read_size = _parent.ReadAt(column.file, aligned_disk_offset, cache.buffer, READ_BUFFER_SIZE);
				if(read_size <= 0) {
					PVLOG_ERROR("PVNRawDiskBackend: Error reading column %d [offset=%d] from disk (%s)\n", col, aligned_disk_offset, strerror(errno));
					return 0;
				}
				cache.first_field = field_index == -1 ? 0 :_parent._indexes.at(field_index, col).field;
				cache.last_field = _parent._indexes.at(field_index+1, col).field-1;
				column.buffer_read = buffer_ptr;
			}

			// Update cache timestamp
			cache.timestamp = tbb::tick_count::now();

			column.buffer_read_ptr = buffer_ptr;

			return field - cache.first_field;
		}

		~PVCachePool()
		{
			for (uint64_t cache_idx = 0; cache_idx < NB_CACHE_BUFFERS; cache_idx++) {
				PVCore::PVAlignedAllocator<char, BUF_ALIGN>().deallocate(_caches[cache_idx].buffer, READ_BUFFER_SIZE);
			}
		}

	private:
		int64_t inline get_index(uint64_t col, uint64_t field)
		{
			index_table_t& indexes = _parent._indexes;
			/*int64_t first = 0;
			int64_t index = 0;
			int64_t last = _parent._indexes_nrows-1;

			if (field >= indexes.at(last, col).field) return last;

			while (first <= last) {
				int index = (first + last) / 2;

				if (field >= indexes.at(index, col).field) {
					if (field < indexes.at(index+1, col).field) {
						return index;
					}
					first = index + 1;
				}
				else if (field < indexes.at(index, col).field) {
					last = index - 1;
					if (last < 0) {
						return 0;
					}
				}
			}*/
			if (_parent._indexes_nrows == 0) {
				return -1;
			}

			for (int64_t index = _parent._indexes_nrows-1; index >= 0; index--) {
				if (indexes.at(index, col).field <= field) {
					return index;
				}
			}
			return -1;
		}

	private:
		struct PVReadCache
		{
			char* buffer = nullptr;
			char* buffer_ptr = nullptr;
			uint64_t column = INVALID;
			tbb::tick_count timestamp = tbb::tick_count::now();
			uint64_t first_field = INVALID;
			uint64_t last_field = INVALID; // This one is *not* included in this cache
		};

	private:
		PVReadCache _caches[NB_CACHE_BUFFERS];
		PVRush::PVNRawDiskBackend<T>& _parent;
	};

private:
	inline PVColumn& get_col(uint64_t col) { return _columns[col]; }

private:
	std::string _nraw_folder;
	uint64_t _num_cols;

	bool _direct_mode = true;

	std::vector<PVColumn> _columns;
	index_table_t _indexes;

	PVCachePool<FilePolicy> _cache_pool;

	const uint64_t _write_buffers_size_pattern[8] = { // too bad we need to specify the size of the array...
		128*1024,
		256*1024,
		512*1024,
		1*1024*1024,
		2*1024*1024,
		4*1024*1024,
		8*1024*1024,
		16*1024*1024
	};
	uint64_t _max_write_size_idx = sizeof(_write_buffers_size_pattern)/sizeof(uint64_t)-1;

	const uint64_t _index_fields_size_pattern[4] = { // too bad we need to specify the size of the array...
		16,
		32,
		64,
		256
	};
	const uint64_t _max_fields_size_idx = sizeof(_index_fields_size_pattern)/sizeof(uint64_t)-1;
	uint64_t _fields_size_idx = 0;
	uint64_t _next_indexes_nrows = _index_fields_size_pattern[0];
	uint64_t _indexes_nrows = 0;

	tbb::tick_count::interval_t _matrix_resize_interval;
	tbb::tick_count::interval_t _search_interval;
};

}

#endif /* PVNRAWDISKBACKEND_H_ */
