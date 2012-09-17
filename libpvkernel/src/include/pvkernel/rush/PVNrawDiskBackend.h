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

constexpr uint64_t BUF_ALIGN = 512;
constexpr uint64_t READ_BUFFER_SIZE = 1024 + BUF_ALIGN;
constexpr uint64_t FIELDS_PER_INDEX = 8192;
constexpr uint64_t NB_CACHE_BUFFERS = 3;
constexpr uint64_t INVALID = UINT64_MAX;

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

	bool Open(std::string const& filename, file_t* file, bool direct=true)
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
		return read(file, buffer, buf_size);
	}

	inline int64_t ReadAt(file_t file, uint64_t offset, void* buffer, uint64_t buf_size)
	{
		int64_t r = lseek(file, offset, SEEK_SET);
		if (r == -1) {
			std::cout << "lseek [offset=" << offset << "]: " << strerror(errno) << std::endl;
		}
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

	void Flush(file_t)
	{
	}

	void Close(file_t file)
	{
		close(file);
	}
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
		return fread(buffer, 1, buf_size, file);
	}

	inline int64_t ReadAt(file_t file, uint64_t offset, void* buffer, uint64_t buf_size)
	{
		fseek(file, offset, SEEK_SET);
		return fread(buffer, 1, buf_size, file);
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
public:
	typedef typename FilePolicy::file_t file_t;
	typedef std::pair<uint64_t, uint64_t> offset_fields_t;
	typedef PVCore::PVMatrix<offset_fields_t, PVRow, PVCol> index_table_t;
	typedef PVNRawDiskBackend<FilePolicy> this_type;

public:
	PVNRawDiskBackend(std::string const& nraw_folder, uint64_t num_cols) :
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
			if(!this->Open(filename.str(), &column.file)) {
				PVLOG_ERROR("PVNRawDiskBackend: Error opening file %s (%s)\n", filename.str().c_str(), strerror(errno));
				return;
			}

			// Create buffer
			uint64_t buffer_size = _buffers_size_pattern[0];
			column.buffer_write = PVCore::PVAlignedAllocator<char, BUF_ALIGN>().allocate(buffer_size);
			column.buffer_write_ptr = column.buffer_write;
			column.buffer_write_end_ptr = column.buffer_write + buffer_size;
			column.field_length = 0; // Or any value grater than 0 to specify a fixed field length;
		}
	}

	uint64_t add(PVCol col_idx, const char* field, uint64_t field_size)
	{
		PVColumn& column = get_col(col_idx);
		field_size += column.end_char();
		uint64_t field_part2_size = 0;
		char* field_part2 = nullptr;
		uint64_t write_size = 0;

		// Index field
		if (column.fields_ignored_size + field_size > READ_BUFFER_SIZE) {
			uint64_t field_offset_in_file = this->Tell(column.file) + (column.buffer_write_ptr - column.buffer_write);
			_indexes.resize(++column.fields_indexed+1, _num_cols);
			_indexes.set_value(column.fields_indexed/*-1*/, col_idx, std::make_pair(field_offset_in_file, column.fields_nb));
			column.fields_ignored_size = 0;
		}
		else {
			column.fields_ignored_size += field_size;
		}
		column.fields_nb++;

		// Fill the buffer with complete field
		if (column.buffer_write_ptr + field_size <= column.buffer_write_end_ptr) {
			memcpy(column.buffer_write_ptr, field, field_size);
			column.buffer_write_ptr += field_size;
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
			write_size = this->Write(column.buffer_write, _buffers_size_pattern[column.buffers_write_size_idx], column.file);
			if(write_size <= 0) {
				PVLOG_ERROR("PVNRawDiskBackend: Error writing column %d to disk (%s)\n", col_idx, strerror(errno));
				return 0;
			}

			uint64_t buffer_max_size = _buffers_size_pattern[column.buffers_write_size_idx];

			// Reallocate a bigger buffer_write and copy the end of splitted field
			if (column.buffers_write_size_idx < _max_size_idx) {
				uint64_t new_buffer_max_size = _buffers_size_pattern[++column.buffers_write_size_idx];
				PVCore::PVAlignedAllocator<char, BUF_ALIGN>().deallocate(column.buffer_write, buffer_max_size);
				column.buffer_write = PVCore::PVAlignedAllocator<char, BUF_ALIGN>().allocate(new_buffer_max_size);
				column.buffer_write_end_ptr = column.buffer_write + new_buffer_max_size;

				memcpy(column.buffer_write, field_part2, field_part2_size);
				column.buffer_write_ptr = column.buffer_write + field_part2_size;
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
		// WARNING: O_DIRECT option prevent to write a buffer of an arbitrary size (must be multiple of 512)

		/*for (uint64_t col_idx = 0 ; col_idx < _num_cols; col_idx++) {
		 	PVColumn& column = get_col(col_idx);
			uint64_t partial_buffer_size = column.buffer_end_ptr - column.buffer_ptr;
			if(!this->Write(column.buffer, partial_buffer_size,  column.file)) {
				PVLOG_ERROR("PVNRawDiskBackend: Error writing column %d to disk (%s)\n", col_idx, strerror(errno));
				return;
			}
		}*/
	}

	char* at(PVRow field, PVCol col)
	{
		PVColumn& column = get_col(col);
		column.last_accessed_field = field;

		uint64_t nb_fields_left = _cache_pool.get_cache(field, col);

		return next(col, nb_fields_left, column.buffer_read);
	}

	inline char* next(uint64_t col)
	{
		PVColumn& column = get_col(col);
		return next(col, 1, column.buffer_read_ptr);
	}

	uint64_t search_in_column(uint64_t col_idx, std::string const& field)
	{
		PVColumn& column = get_col(col_idx);

		uint64_t chunk_size = _buffers_size_pattern[_max_size_idx];
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

	void store_index_to_disk()
	{
		char* data = (char*) _indexes.get_data();
		file_t file;
		std::stringstream filename;
		filename << _nraw_folder << "/nraw.idx";
		this->Open(filename.str(), &file, false);
		uint64_t size = _indexes.get_ncols()*_indexes.get_nrows()*sizeof(offset_fields_t);
		int64_t write_size = this->Write(data, size, file);
		if(write_size <= 0) {
			PVLOG_ERROR("PVNRawDiskBackend: Error writing index to disk [size=%d] (%s)\n", size, strerror(errno));
			return;
		}
		this->Close(file);
		std::cout << "nrows=" << _indexes.get_nrows() << " ncols=" << _indexes.get_ncols() << std::endl;
	}

	void load_index_from_disk(uint64_t nrows, uint64_t ncols)
	{
		std::stringstream filename;
		filename << _nraw_folder << "/nraw.idx";
		file_t file;
		this->Open(filename.str(), &file, false);
		uint64_t size = nrows*ncols*sizeof(offset_fields_t);
		_indexes.resize(nrows, ncols);
		char* data = (char*) _indexes.get_data();
		int64_t read_size = this->Read(file, data, size);
		if(read_size <= 0) {
			PVLOG_ERROR("PVNRawDiskBackend: Error reading index from disk (%s)\n", strerror(errno));
			return;
		}
		this->Close(file);
	}

	~PVNRawDiskBackend()
	{
		// Close files and deallocate all buffers
		for (uint64_t col_idx = 0 ; col_idx < _num_cols; col_idx++) {
			PVColumn& column = get_col(col_idx);
			this->Close(column.file);
			PVCore::PVAlignedAllocator<char, BUF_ALIGN>().deallocate(column.buffer_write, _buffers_size_pattern[column.buffers_write_size_idx]);
		}
	}

private:
	inline char* next(uint64_t col, uint64_t nb_fields, char* buffer)
	{
		PVColumn& column = get_col(col);

		// Extract proper field from buffer
		char* buffer_ptr = buffer;
		if (column.field_length > 0) {
			buffer_ptr += ((column.last_accessed_field - nb_fields) * column.field_length);
		}
		else {
			char* end_field_ptr = nullptr;
			uint64_t size_to_read = READ_BUFFER_SIZE;
			for (uint64_t i = 0; i < nb_fields; i++) {
				end_field_ptr = (char*) memchr(buffer_ptr, '\0', size_to_read);
				if (end_field_ptr == nullptr) {
					buffer_ptr = nullptr;
					break;
				}
				uint64_t field_length = (end_field_ptr - buffer_ptr);
				size_to_read -= field_length;
				buffer_ptr += (field_length+1);
			}
		}

		column.buffer_read_ptr = buffer_ptr;
		return buffer_ptr;
	}


private:
	struct PVColumn
	{
	public:
		bool end_char() { return field_length == 0; }

	public:
		file_t file = 0;

		char* buffer_write = nullptr;
		char* buffer_write_ptr = nullptr;
		char* buffer_write_end_ptr = nullptr;
		uint64_t buffers_write_size_idx = 0;

		uint64_t field_length = 0;
		uint64_t fields_nb = 0;
		uint64_t fields_ignored_size = 0;
		uint64_t fields_indexed = 0;

		uint64_t last_accessed_field = 0;
		char* buffer_read = nullptr;
		char* buffer_read_ptr = nullptr;
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
				uint64_t field_index = get_index(col, field);
				uint64_t disk_offset = _parent._indexes.at(field_index, col).first;
				uint64_t aligned_disk_offset = (disk_offset / BUF_ALIGN) * BUF_ALIGN;
				int64_t read_size = _parent.ReadAt(column.file, aligned_disk_offset, cache.buffer, READ_BUFFER_SIZE);
				if(read_size <= 0) {
					PVLOG_ERROR("PVNRawDiskBackend: Error reading column %d [offset=%d] from disk (%s)\n", col, aligned_disk_offset, strerror(errno));
					return 0;
				}
				column.buffer_read = cache.buffer;
				buffer_ptr += (disk_offset - aligned_disk_offset);
				cache.first_field = _parent._indexes.at(field_index, col).second;
				cache.last_field = _parent._indexes.at(field_index+1, col).second;
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
		uint64_t inline get_index(uint64_t col, uint64_t field)
		{
			index_table_t& indexes = _parent._indexes;
			uint64_t first = 0;
			uint64_t index = 0;
			uint64_t last = indexes.get_height()-1;

			if (field >= indexes.at(last, col).second) return last;

			while (first <= last) {
				int index = (first + last) / 2;

				if (field >= indexes.at(index, col).second) {
					if (field < indexes.at(index+1, col).second) {
						return index;
					}
					first = index + 1;
				}
				else if (field < indexes.at(index, col).second) {
					last = index - 1;
				}
			}

			return index;
		}

	private:
		struct PVReadCache
		{
			char* buffer = nullptr;
			char* buffer_ptr = nullptr;
			uint64_t column = INVALID;
			tbb::tick_count timestamp = tbb::tick_count::now();
			uint64_t first_field = INVALID;
			uint64_t last_field = INVALID;
		};

	private:
		PVReadCache _caches[NB_CACHE_BUFFERS];
		PVRush::PVNRawDiskBackend<T>& _parent;
	};

private:
	inline PVColumn& get_col(uint64_t col) { return _columns[col]; }

private:
	const std::string& _nraw_folder;
	uint64_t _num_cols;

	std::vector<PVColumn> _columns;
	index_table_t _indexes;

	PVCachePool<FilePolicy> _cache_pool;

	uint64_t _buffers_size_pattern[8] = { // too bad we need to specify the size of the array...
		128*1024,
		256*1024,
		512*1024,
		1*1024*1024,
		2*1024*1024,
		4*1024*1024,
		8*1024*1024,
		16*1024*1024
	};
	uint64_t _max_size_idx = sizeof(_buffers_size_pattern)/sizeof(uint64_t)-1;
};

}

#endif /* PVNRAWDISKBACKEND_H_ */
