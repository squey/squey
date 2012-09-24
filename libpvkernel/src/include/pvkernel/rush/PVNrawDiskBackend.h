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

#include <tbb/concurrent_queue.h>
#include <tbb/pipeline.h>
#include <tbb/tick_count.h>

#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/PVAllocators.h>
#include <pvkernel/core/PVMatrix.h>
#include <pvkernel/core/PVByteVisitor.h>

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

	bool Open(std::string const& filename, file_t* file, bool direct = true, bool trunc = false)
	{
		int64_t flags = O_RDWR | O_CREAT;
		if (direct) {
			flags |= O_DIRECT;
		}
		if (trunc) {
			flags |= O_TRUNC;
		}
		*file = open(filename.c_str(), flags, 0640);
		return *file != -1;
	}

	inline int64_t Write(const void* content, uint64_t buf_size, file_t file)
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
		off_t ret;
		if ((ret = lseek(file, offset, SEEK_SET)) == (off_t)-1) {
			PVLOG_ERROR("Unable to seek into file\n");
			return -1;
		}
		return ret;
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

	void Truncate(file_t& file, off_t l)
	{
		ftruncate(file, l);
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

	inline int64_t Write(const void* content, uint64_t buf_size, file_t file)
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
class PVNrawDiskBackend: private FilePolicy
{
	static constexpr uint64_t BUF_ALIGN = 512;
	static constexpr uint64_t READ_BUFFER_SIZE = 2*1024*1024;
	static constexpr uint64_t NB_CACHE_BUFFERS = 3;
	static constexpr uint64_t INVALID = UINT64_MAX;
	static constexpr size_t   SERIAL_READ_BUFFER_SIZE = 2*1024*1024;

private:
	struct offset_fields_t
	{
		// At the offset `offset', the field number `field' starts.
		uint64_t offset = 0;
		uint64_t field = 0;
	};

public:
	class tbb_chunks_t
	{
		struct _chunk_t
		{
			char buf[SERIAL_READ_BUFFER_SIZE];
			size_t start_field;
			size_t end_field;
			size_t size_data;
			size_t __padding;
		};

		static_assert(sizeof(_chunk_t) % 16 == 0, "chunk_t isn't a multiple of 16 !\n");

	public:
		typedef _chunk_t chunk_t;

	public:
		tbb_chunks_t(size_t n):
			_n(n)
		{
			_chunks = PVCore::PVAlignedAllocator<chunk_t, 16>().allocate(n);
			_queue.set_capacity(n);
			for (size_t i = 0; i < n; i++) {
				_queue.try_push(&_chunks[i]);
			}
		}

		~tbb_chunks_t()
		{
			PVCore::PVAlignedAllocator<chunk_t, 16>().deallocate(_chunks, _n);
		}

		chunk_t* get_chunk()
		{
			chunk_t* ret;
			_queue.pop(ret);
			assert(((uintptr_t)ret % 16) == 0);
			return ret;
		}

		void release_chunk(chunk_t* const c)
		{
			assert(((uintptr_t)c - (uintptr_t)_chunks) % sizeof(chunk_t) == 0);
			assert(std::distance(_chunks, c) < _n);
			_queue.push(c);
		}

		size_t size() const { return _n; }

	private:
		chunk_t* _chunks;
		size_t   _n;
		tbb::concurrent_bounded_queue<chunk_t*> _queue;
	};


public:
	typedef typename FilePolicy::file_t file_t;
	//typedef std::pair<uint64_t, uint64_t> offset_fields_t;
	typedef PVCore::PVMatrix<offset_fields_t, PVRow, PVCol> index_table_t;
	typedef PVNrawDiskBackend<FilePolicy> this_type;

public:
	PVNrawDiskBackend():
		_cache_pool(*this),
		_nrows(0),
		_serial_read_buffer(nullptr),
		_chunks(48)
	{
	}

	~PVNrawDiskBackend()
	{
		if (_serial_read_buffer) {
			PVCore::PVAlignedAllocator<char, BUF_ALIGN>().deallocate(_serial_read_buffer, SERIAL_READ_BUFFER_SIZE);
		}
		// Close files
		for (uint64_t col_idx = 0 ; col_idx < get_number_cols(); col_idx++) {
			PVColumn& column = get_col(col_idx);
			this->Close(column.file);
		}
	}

	void init(const char* nraw_folder, const uint64_t num_cols)
	{
		_nraw_folder = nraw_folder;
		_columns.reserve(num_cols);
		for (uint64_t col = 0 ; col < num_cols ; col++) {
			_columns.emplace_back();
			PVColumn& column = _columns.back();

			// Open file
			column.filename = std::move(get_disk_column_file(col));
			if(!this->Open(column.filename, &column.file, true, true)) {
				PVLOG_ERROR("PVNrawDiskBackend: Error opening file %s (%s)\n", column.filename.c_str(), strerror(errno));
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

		_serial_read_buffer = PVCore::PVAlignedAllocator<char, BUF_ALIGN>().allocate(SERIAL_READ_BUFFER_SIZE);
	}

public:
	uint64_t add(PVCol col_idx, const char* field, const uint64_t field_size)
	{
		PVColumn& column = get_col(col_idx);
		const uint64_t written_field_size = field_size + column.end_char();
		uint64_t field_part2_size = 0;
		char* field_part2 = nullptr;
		uint64_t write_size = 0;

		// Index field
		if (column.fields_ignored_size + written_field_size > READ_BUFFER_SIZE) {
			assert(column.buffer_write <= column.buffer_write_ptr);
			uint64_t field_offset_in_file = this->Tell(column.file) + ((uintptr_t)column.buffer_write_ptr - (uintptr_t)column.buffer_write) ;

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
			assert(column.buffer_write_ptr <= column.buffer_write_end_ptr);
		}
		// Fill the buffer_write with splitted field
		else {
			assert((uintptr_t)column.buffer_write_end_ptr >= (uintptr_t)column.buffer_write_ptr);
			uint64_t field_part1_size = column.buffer_write_end_ptr - column.buffer_write_ptr;
			memcpy(column.buffer_write_ptr, field, field_part1_size);
			field_part2 = (char *)(field + field_part1_size);
			field_part2_size = field_size - field_part1_size;
			column.buffer_write_ptr += field_part1_size;
			assert(column.buffer_write_ptr == column.buffer_write_end_ptr);
		}

		// Write buffer_write to disk
		if (column.buffer_write_ptr == column.buffer_write_end_ptr) {
			write_size = this->Write(column.buffer_write, _write_buffers_size_pattern[column.buffers_write_size_idx], column.file);
			if(write_size <= 0) {
				PVLOG_ERROR("PVNrawDiskBackend: Error writing column %d to disk (%s)\n", col_idx, strerror(errno));
				return 0;
			}

			uint64_t buffer_max_size = _write_buffers_size_pattern[column.buffers_write_size_idx];

			// Reallocate a bigger buffer_write and copy the end of splitted field
			if (column.buffers_write_size_idx < _max_write_size_idx) {
				uint64_t new_buffer_max_size = _write_buffers_size_pattern[++column.buffers_write_size_idx];
				PVCore::PVAlignedAllocator<char, BUF_ALIGN>().deallocate(column.buffer_write, buffer_max_size);
				column.buffer_write = PVCore::PVAlignedAllocator<char, BUF_ALIGN>().allocate(new_buffer_max_size);
				column.buffer_write_ptr = column.buffer_write;
				column.buffer_write_end_ptr = column.buffer_write + new_buffer_max_size;
			}
			// Recycle previously allocated buffer_write
			else {
				column.buffer_write_ptr = column.buffer_write;
			}

			if (field_part2_size > 0) {
				memcpy(column.buffer_write, field_part2, field_part2_size);
				column.buffer_write[field_part2_size] = 0;
				column.buffer_write_ptr = column.buffer_write + field_part2_size + 1;
				assert(column.buffer_write_ptr <= column.buffer_write_end_ptr);
			}
		}
		_nrows = std::max(_nrows, column.fields_nb);

		return write_size;
	}

	void flush()
	{
		// direct mode prevents to write a buffer of an arbitrary size
		set_direct_mode(false);

		for (uint64_t col_idx = 0 ; col_idx < get_number_cols(); col_idx++) {
		 	PVColumn& column = get_col(col_idx);
			uint64_t partial_buffer_size = column.buffer_write_ptr - column.buffer_write;
			if (partial_buffer_size > 0) {
				if(!this->Write(column.buffer_write, partial_buffer_size, column.file)) {
					PVLOG_ERROR("PVNrawDiskBackend: Error writing column %d to disk (%s)\n", col_idx, strerror(errno));
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

	/*
	 * AG: we need to be able to go to the next chunk when this is necessary!
	inline const char* next(uint64_t col, size_t& size_ret)
	{
		PVColumn& column = get_col(col);
		assert(column.buffer_read != nullptr);
		return next(col, 1, column.buffer_read_ptr, size_ret);
	}*/

	template <typename F>
	bool visit_column(uint64_t const col_idx, F const& f)
	{
		set_direct_mode(false);

		// Sequential version
		PVColumn& column = get_col(col_idx);
		this->Seek(column.file, 0);

		ssize_t read_size;
		//size_t cur_idx = 0;
		size_t cur_field = 0;
		char* cur_rbuf = _serial_read_buffer;
		while (true) {
			read_size = this->Read(column.file, cur_rbuf, SERIAL_READ_BUFFER_SIZE-std::distance(_serial_read_buffer, cur_rbuf));
			if (read_size <= 0) {
				break;
			}
			size_t buf_size = std::distance(_serial_read_buffer, cur_rbuf) + read_size;
			size_t processed_size = visit_column_process_chunk_sse(cur_field, _nrows-1, _serial_read_buffer, buf_size, f);
			if (cur_field == _nrows) {
				return true;
			}
			/*
			else {
				// Go throught the buffer index by index
				char* cur_buf = _serial_read_buffer;
				while ((read_size > 0) && (cur_idx < _indexes_nrows)) {
					const offset_fields_t& field_end = index_col.at(cur_idx);
					processed_size = visit_column_process_chunk(cur_field, field_end.field, cur_buf, size_chunk, read_size);
					assert(cur_field <= field_end.field);
					if (cur_field == field_end.field) {
						cur_idx++;
					}

					cur_buf += processed_size;
					assert(read_size >= processed_size);
					read_size -= processed_size;
				}
			}*/
			assert(processed_size <= buf_size);
			if (processed_size < buf_size) {
				const size_t diff = buf_size-processed_size;
				memmove(_serial_read_buffer, _serial_read_buffer + processed_size, diff);
				cur_rbuf = _serial_read_buffer + diff;
			}
		}
		return true;
	}

	template <typename F>
	bool visit_column2(uint64_t const col_idx, F const& f)
	{
		set_direct_mode(false);

		// Sequential version
		PVColumn& column = get_col(col_idx);
		this->Seek(column.file, 0);

		ssize_t read_size;
		size_t prev_off = 0;
		size_t cur_field = 0;
		typename index_table_t::column index_col = _indexes.get_col(col_idx);
		for (size_t i = 0; i < _indexes_nrows; i++) {
			offset_fields_t const& off_field = index_col.at(i);
			const size_t off = off_field.offset;
			const size_t end_field = off_field.field;
			const size_t diff_off = off-prev_off;
			assert(diff_off <= SERIAL_READ_BUFFER_SIZE);
			read_size = this->Read(column.file, _serial_read_buffer, diff_off);
			if (read_size != diff_off) {
				assert(false);
				return false;
			}
			const size_t processed_size = visit_column_process_chunk_sse(cur_field, end_field-1, _serial_read_buffer, read_size, f);
			if ((processed_size != read_size) || (cur_field != end_field)) {
				assert(false);
				return false;
			}
			prev_off = off;
		}
		// Finish off !
		// We have an index at most every BUFFER_READ, so as SERIAL_READ_BUFFER_SIZE > BUFFER_READ, only one read is now
		// necessary !
		read_size = this->Read(column.file, _serial_read_buffer, SERIAL_READ_BUFFER_SIZE);
		visit_column_process_chunk_sse(cur_field, _nrows-1, _serial_read_buffer, read_size, f);

		return cur_field == _nrows;
	}

	template <typename F>
	bool visit_column_tbb(uint64_t const col_idx, F const& f)
	{
		set_direct_mode(false);

		// TBB version
		PVColumn& column = get_col(col_idx);
		this->Seek(column.file, 0);

		size_t prev_off = 0;
		size_t cur_field = 0;
		typename index_table_t::column index_col = _indexes.get_col(col_idx);
		size_t cur_idx = 0;
		tbb::parallel_pipeline(_chunks.size(),
			tbb::make_filter<void, typename tbb_chunks_t::chunk_t*>(tbb::filter::serial_in_order,
				[&](tbb::flow_control& fc) -> typename tbb_chunks_t::chunk_t*
				{
					if (cur_idx >= _indexes_nrows) {
						fc.stop();
						return nullptr;
					}
					typename tbb_chunks_t::chunk_t* chunk = this->_chunks.get_chunk();
					offset_fields_t const& off_field = index_col.at(cur_idx);
					cur_idx++;
					const size_t off = off_field.offset;
					const size_t end_field = off_field.field;
					const size_t diff_off = off-prev_off;
					assert(diff_off <= SERIAL_READ_BUFFER_SIZE);
					const ssize_t read_size = this->Read(column.file, chunk->buf, diff_off);
					if (read_size != diff_off) {
						assert(false);
						fc.stop();
						return nullptr;
					}
					chunk->size_data = diff_off;
					chunk->start_field = cur_field;
					chunk->end_field = end_field;
					cur_field = end_field;
					prev_off = off;
					return chunk;
				}) &

			tbb::make_filter<typename tbb_chunks_t::chunk_t*, typename tbb_chunks_t::chunk_t*>(tbb::filter::parallel,
				[&](typename tbb_chunks_t::chunk_t* c)
				{
#ifdef NDEBUG
					this->visit_column_process_chunk_sse(c->start_field, c->end_field-1, &c->buf[0], c->size_data, f);
#else
					const size_t processed_size = this->visit_column_process_chunk_sse(c->start_field, c->end_field-1, &c->buf[0], c->size_data, f);
					assert(!((processed_size != c->size_data) || (c->start_field != c->end_field)));
#endif
					return c;
				}) &

			tbb::make_filter<typename tbb_chunks_t::chunk_t*, void>(tbb::filter::serial_out_of_order,
				[&](typename tbb_chunks_t::chunk_t* c)
				{
					this->_chunks.release_chunk(c);
				})
		);

		// Finish off !
		// We have an index at most every BUFFER_READ, so as SERIAL_READ_BUFFER_SIZE > BUFFER_READ, only one read is now
		// necessary !
		size_t read_size = this->Read(column.file, _serial_read_buffer, SERIAL_READ_BUFFER_SIZE);
		visit_column_process_chunk_sse(cur_field, _nrows-1, _serial_read_buffer, read_size, f);

		return cur_field == _nrows;
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
		if (_direct_mode == direct) {
			return;
		}

		for (int col = 0 ; col < get_number_cols() ; col++) {
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
		this->Open(get_disk_index_file(), &file, false);
		this->Write(&_nrows, sizeof(size_t), file);
		uint64_t size = get_number_cols() * _indexes_nrows * sizeof(offset_fields_t);
		if (size > 0) {
			int64_t write_size = this->Write(data, size, file);
			if(write_size <= 0) {
				PVLOG_ERROR("PVNrawDiskBackend: Error writing index to disk [size=%d] (%s)\n", size, strerror(errno));
				return;
			}
		}
		this->Close(file);
	}

	void load_index_from_disk()
	{
		set_direct_mode(false);
		file_t file;
		this->Open(get_disk_index_file(), &file, false);
		if (this->Read(&_nrows, sizeof(size_t), file) != sizeof(size_t)) {
			PVLOG_ERROR("PVNrawDiskBackend: Error reading index from disk (%s)\n", strerror(errno));
			return;
		}
		uint64_t size = this->Size(file);
		_indexes_nrows = size / get_number_cols() / sizeof(offset_fields_t);
		std::cout << "size=" << size << std::endl;
		_indexes.resize(_indexes_nrows, get_number_cols());
		char* data = (char*) _indexes.get_data();
		int64_t read_size = this->Read(file, data, size);
		std::cout << "read_size=" << read_size << std::endl;
		if(read_size <= 0) {
			PVLOG_ERROR("PVNrawDiskBackend: Error reading index from disk (%s)\n", strerror(errno));
			return;
		}
		this->Close(file);
	}

	void clear()
	{
		_indexes.clear();
		unlink(get_disk_index_file().c_str());
		for (uint64_t c = 0 ; c < get_number_cols() ; c++) {
			PVColumn& nraw_c = _columns[c];
			this->Truncate(nraw_c.file, 0);
			
			nraw_c.buffer_write_ptr = nraw_c.buffer_write;
			nraw_c.field_length = 0; // Or any value grater than 0 to specify a fixed field length;
		}

		_next_indexes_nrows = _index_fields_size_pattern[0];
		_indexes.resize_nrows(_next_indexes_nrows);
		_fields_size_idx = 0;
		_next_indexes_nrows += _index_fields_size_pattern[++_fields_size_idx];
		_nrows = 0;
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
			for (uint64_t c=0; c < get_number_cols(); c++) {
				PVLOG_INFO("index[%d][%d] = %d (offset), %d (field)\n", r, c, _indexes.at(r, c).offset, _indexes.at(r, c).field);
			}
		}
	}

	size_t get_number_cols() const { return _columns.size(); }

private:
	template <typename F>
	size_t visit_column_process_chunk(size_t& field_start, size_t const field_end, const char* buf, size_t size_buf, F const& f)
	{
		// field_end is inclusive. field_start will be set to the field following the last *full* field that has been read. It will always be <= field_end.
		// returns the number of bytes *processed*, *including* the '\0' byte of the last processed field
		
		size_t idx_start_field = 0;
		size_t i;
		for (i = 0; i < size_buf; i++) {
			if (buf[i] == '\0') {
				f(field_start, &buf[idx_start_field], i-idx_start_field);
				idx_start_field = i+1;
				field_start++;
				if (field_start > field_end) {
					break;
				}
			}
		}
		return idx_start_field;
	}

	template <typename F>
	size_t visit_column_process_chunk_sse(size_t& field_start, size_t const field_end, const char* buf, size_t size_buf, F const& f)
	{
		// field_end is inclusive. field_start will be set to the field following the last *full* field that has been read. It will always be <= field_end.
		// returns the number of bytes *processed*, *including* the '\0' byte of the last processed field
		
		size_t idx_start_field = 0;
		size_t i;
		const size_t size_buf_sse = (size_buf>>4)<<4;
		const __m128i sse_zero = _mm_setzero_si128();
		const __m128i sse_ff = _mm_set1_epi32(0xffffffff);
		for (i = 0; i < size_buf_sse; i += 16) {
			const __m128i sse_buf = _mm_load_si128((const __m128i*)&buf[i]);
			const __m128i sse_cmp = _mm_cmpeq_epi8(sse_buf, sse_zero);
			if (!_mm_testz_si128(sse_cmp, sse_ff)) {
				PVCore::PVByteVisitor::visit_bytes(sse_cmp,
					[&](size_t b)
					{
						f(field_start, &buf[idx_start_field], b-idx_start_field);
						idx_start_field = b+1;
						field_start++;
					},
					i
				);
				if (field_start > field_end) {
					return idx_start_field;
				}
#if 0
				const uint64_t p0 = _mm_extract_epi64(sse_cmp, 0);
				const uint64_t p1 = _mm_extract_epi64(sse_cmp, 1);
				for (unsigned int j = 0; j < 8; j++) {
					if (p0 & (0xFFULL << (j*8))) {
						f(field_start, &buf[idx_start_field], j+i-idx_start_field);
						idx_start_field = i+1+j;
						field_start++;
						if (field_start > field_end) {
							break;
						}
					}
				}
				for (unsigned int j = 0; j < 8; j++) {
					if (p1 & (0xFFULL << (j*8))) {
						f(field_start, &buf[idx_start_field], 8+j+i-idx_start_field);
						idx_start_field = i+1+j+8;
						field_start++;
						if (field_start > field_end) {
							break;
						}
					}
				}
#endif
			}
		}
		for (; i < size_buf; i++) {
			if (buf[i] == '\0') {
				f(field_start, &buf[idx_start_field], i-idx_start_field);
				idx_start_field = i+1;
				field_start++;
				if (field_start > field_end) {
					break;
				}
			}
		}
		return idx_start_field;
	}

	std::string get_disk_index_file()
	{
		std::stringstream filename;
		filename << _nraw_folder << "/" << INDEX_FILENAME;
		return std::move(filename.str());
	}

	std::string get_disk_column_file(uint64_t col)
	{
		assert(col < get_number_cols());
		std::stringstream filename;
		filename << _nraw_folder << "/column_" << col;
		return std::move(filename.str());
	}

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

		PVCachePool(PVRush::PVNrawDiskBackend<T>& parent) : _parent(parent)
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
					PVLOG_ERROR("PVNrawDiskBackend: Error reading column %d [offset=%d] from disk (%s)\n", col, aligned_disk_offset, strerror(errno));
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
		PVRush::PVNrawDiskBackend<T>& _parent;
	};

private:
	inline PVColumn& get_col(uint64_t col) { assert(col < _columns.size()); return _columns[col]; }

private:
	std::string _nraw_folder;
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

	size_t _nrows;
	char* _serial_read_buffer;

	tbb_chunks_t _chunks;
};

}

#endif /* PVNRAWDISKBACKEND_H_ */
