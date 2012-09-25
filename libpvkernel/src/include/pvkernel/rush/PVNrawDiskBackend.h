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

	static bool Open(std::string const& filename, file_t* file, bool direct = true, bool trunc = false)
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

	static inline int64_t Write(const void* content, uint64_t buf_size, file_t file)
	{
		return write(file, content, buf_size);
	}

	static inline int64_t Read(file_t file, void* buffer,  uint64_t buf_size)
	{
		int64_t r = read(file, buffer, buf_size);
		return r;
	}

	static inline int64_t ReadAt(file_t file, uint64_t offset, void* buffer, uint64_t buf_size)
	{
		lseek(file, offset, SEEK_SET);
		return Read(file, buffer, buf_size);
	}

	static inline int64_t Seek(file_t file, int64_t offset)
	{
		off_t ret;
		if ((ret = lseek(file, offset, SEEK_SET)) == (off_t)-1) {
			PVLOG_ERROR("Unable to seek into file\n");
			return -1;
		}
		return ret;
	}

	static inline int64_t Tell(file_t file)
	{
		return lseek(file, 0, SEEK_CUR);
	}

	static inline uint64_t Size(file_t file)
	{
		int64_t pos = Tell(file);
		uint64_t size = lseek(file, 0, SEEK_END);
		lseek(file, pos, SEEK_SET);
		return size;
	}

	static void Flush(file_t)
	{
	}

	static void Close(file_t file)
	{
		close(file);
	}

	static void Truncate(file_t& file, off_t l)
	{
		ftruncate(file, l);
	}
};

struct BufferedFilePolicy
{
	typedef FILE* file_t;

	static bool Open(std::string const& filename, file_t* file)
	{
		*file = fopen(filename.c_str(), "rw");
		return *file != nullptr;
	}

	static inline int64_t Write(const void* content, uint64_t buf_size, file_t file)
	{
		return fwrite(content, buf_size, 1, file);
	}

	static inline int64_t Read(file_t file, void* buffer, uint64_t buf_size)
	{
		int64_t r = fread(buffer, 1, buf_size, file);
		return r;
	}

	static inline int64_t ReadAt(file_t file, uint64_t offset, void* buffer, uint64_t buf_size)
	{
		fseek(file, offset, SEEK_SET);
		return Read(file, buffer, buf_size);
	}

	static inline int64_t Seek(file_t file, int64_t offset)
	{
		return fseek(file, offset, SEEK_CUR);
	}

	static inline int64_t Tell(file_t file)
	{
		return ftell(file);
	}

	static void Flush(file_t file)
	{
		fflush(file);
	}

	static void Close(file_t file)
	{
		fclose(file);
	}
};

class PVNrawDiskBackend: private RawFilePolicy
{
	static constexpr uint64_t BUF_ALIGN = 512;
	static constexpr uint64_t READ_BUFFER_SIZE = 256*1024;
	static constexpr uint64_t NB_CACHE_BUFFERS = 10;
	static constexpr uint64_t INVALID = std::numeric_limits<uint64_t>::max();
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
		tbb_chunks_t(size_t n);
		~tbb_chunks_t();

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
			assert(std::distance(_chunks, c) < (ssize_t) _n);
			_queue.push(c);
		}

		size_t size() const { return _n; }

	private:
		chunk_t* _chunks;
		size_t   _n;
		tbb::concurrent_bounded_queue<chunk_t*> _queue;
	};


public:
	typedef RawFilePolicy file_policy_t;
	typedef typename file_policy_t::file_t file_t;
	//typedef std::pair<uint64_t, uint64_t> offset_fields_t;
	typedef PVCore::PVMatrix<offset_fields_t, PVRow, PVCol> index_table_t;
	typedef PVNrawDiskBackend this_type;

public:
	PVNrawDiskBackend();
	~PVNrawDiskBackend();

public:
	void init(const char* nraw_folder, const uint64_t num_cols);
	void set_direct_mode(bool direct = true);
	void clear();

	void store_index_to_disk();
	void load_index_from_disk();
public:
	uint64_t add(PVCol col_idx, const char* field, const uint64_t field_size);
	void flush();

	inline const char* at(PVRow field, PVCol col, size_t& size_ret)
	{
		PVColumn& column = get_col(col);
		uint64_t nb_fields_left = _cache_pool.get_cache(field, col);
		return next(col, nb_fields_left, column.buffer_read, size_ret);
	}

	size_t get_number_cols() const { return _columns.size(); }

	/*
	 * AG: we need to be able to go to the next chunk when this is necessary!
	inline const char* next(uint64_t col, size_t& size_ret)
	{
		PVColumn& column = get_col(col);
		assert(column.buffer_read != nullptr);
		return next(col, 1, column.buffer_read_ptr, size_ret);
	}*/

public:
	void print_stats();
	void print_indexes();

public:
	uint64_t search_in_column(uint64_t col_idx, std::string const& field);

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
		for (size_t i = 0; i < column.fields_indexed; i++) {
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
					if (cur_idx >= column.fields_indexed) {
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
					//assert(!((processed_size != c->size_data) || (c->start_field != c->end_field)));
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

private:
	std::string get_disk_index_file() const;
	std::string get_disk_column_file(uint64_t col) const;
 
	char* next(uint64_t col, uint64_t nb_fields, char* buffer, size_t& size_ret);

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

	class PVCachePool: private file_policy_t
	{
	public:

		PVCachePool(PVRush::PVNrawDiskBackend& parent);
		~PVCachePool();

	public:
		uint64_t get_cache(uint64_t field, uint64_t col);

	private:
		int64_t get_index(uint64_t col, uint64_t field);

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
		PVRush::PVNrawDiskBackend& _parent;
	};

private:
	inline PVColumn& get_col(uint64_t col) { assert(col < _columns.size()); return _columns[col]; }

private:
	std::string _nraw_folder;
	bool _direct_mode = true;

	std::vector<PVColumn> _columns;
	index_table_t _indexes;

	PVCachePool _cache_pool;

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
