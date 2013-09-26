/**
 * \file PVNrawDiskBackend.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef PVNRAWDISKBACKEND_H_
#define PVNRAWDISKBACKEND_H_

#include <fcntl.h>
#include <sys/stat.h>

#include <array>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>

#include <tbb/atomic.h>
#include <tbb/concurrent_queue.h>
#include <tbb/pipeline.h>
#include <tbb/task.h>
#include <tbb/tick_count.h>
#include <tbb/enumerable_thread_specific.h>

#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/PVAllocators.h>
#include <pvkernel/core/PVMatrix.h>
#include <pvkernel/core/PVByteVisitor.h>
#include <pvkernel/core/PVSelBitField.h>
#include <pvkernel/core/string_tbb.h>
#include <pvkernel/core/PVHardwareConcurrency.h>
#include <pvkernel/core/PVUtils.h>

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

/**
 * \class PVNrawDiskBackend
 *
 * \note This policy based class is handling the storage of the Nraw on disk.
 *
 */
class PVNrawDiskBackend: private RawFilePolicy
{
public:
	static constexpr uint64_t BUF_ALIGN = 512;
	static constexpr uint64_t READ_BUFFER_SIZE = 4*1024;
	static constexpr uint64_t NB_CACHE_BUFFERS = 10;
	static constexpr uint64_t INVALID = std::numeric_limits<uint64_t>::max();
	static constexpr size_t SERIAL_READ_BUFFER_SIZE = 2*1024*1024;

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
			//ret = _queue.pop();
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
	typedef PVCore::PVMatrix<offset_fields_t, PVRow, PVCol> index_table_t;
	typedef PVNrawDiskBackend this_type;

	// unique values
	typedef std::string_tbb unique_values_key_t;
	typedef size_t unique_values_value_t;
	typedef std::pair<unique_values_key_t, unique_values_value_t> unique_values_container_t;
	typedef std::unordered_map<unique_values_key_t, unique_values_value_t> unique_values_t;

	// count by
	typedef std::string_tbb count_by_key_t;
	typedef size_t count_by_value_t;
	typedef std::unordered_map<count_by_key_t, count_by_value_t> count_by_unique_values_t;
	typedef std::unordered_map<count_by_key_t, count_by_unique_values_t> count_by_t;

public:
	PVNrawDiskBackend();
	~PVNrawDiskBackend();

public:
	/*! \brief Initialize the Nraw disk backend.
	 *
	 *  \param[in] nraw_folder Path to the existing nraw folder on disk.
	 *  \param[in] num_cols The number of columns of the Nraw.
	 */
	void init(const char* nraw_folder, const uint64_t num_cols);

	/*! \brief Enable or disable direct mode that bypass system cache in order to enhance performances.
	 *
	 *  \param[in] direct Specify if direct mode must be enabled or not.
	 *
	 *  \note /!\ Direct mode needs the buffer address to be aligned on 512 and the buffer size to be a multiple of 512.
	 */
	void set_direct_mode(bool direct = true);

	void clear();
	void clear_and_remove();

	/*! \brief Store the columns indexation files to disk.
     */
	void store_index_to_disk();

	/*! \brief Load the columns indexation files from disk.
     */
	void load_index_from_disk();

public:
	/*! \brief Add a field to the end of a given column.
	 *
	 *  \param[in] col_idx The column index.
	 *  \param[in] field The field buffer.
	 *  \param[in] size_ret The field buffer size.
	 *
	 *  \return The number of bytes actually written.
	 */
	uint64_t add(PVCol col_idx, const char* field, const uint64_t field_size);

	/*! \brief Flush the part of the columns remaining in memory onto disk.
	 */
	void flush();

	/*! \brief Returns the string located at a given row/column.
	 *
	 *  \param[in] field The string row.
	 *  \param[in] col The string column.
	 *  \param[out] size_ret The string size.
	 *
	 *  \note This method uses a cache pool to enhance performances.
	 *        Therefore the returned string is only valid as long as the cache is still valid.
	 *        The last returned row index is kept in memory in order to optimize sequencial access.
	 *
	 *  \remark This method is *not* thread-sage
	 *
	 *  \return A pointer to the buffer.
	 */
	inline const char* at(PVRow field, PVCol col, size_t& size_ret)
	{
		PVColumn& column = get_col(col);
		uint64_t nb_fields_left = _cache_pool.get_cache(field, col);
		return next(col, nb_fields_left, column.buffer_read_ptr, size_ret);
	}

	inline const char* at_no_cache(PVRow field, PVCol col, size_t& size_ret) const
	{
		PVColumn const& column = get_col(col);
		return at_no_cache(field, col, size_ret, &column.read_buffer_tls.local()[0]);
	}

	const char* at_no_cache(PVRow field, PVCol col, size_t& size_ret, char* read_buffer) const;

	/*! \brief Returns the number of columns.
	 */
	size_t get_number_cols() const { return _columns.size(); }

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
		file_t& read_file = column.read_file_tls.local();
		if (unlikely(!read_file)) {
			this->Open(column.filename, &read_file, _direct_mode);
		}
		this->Seek(read_file, 0);

		ssize_t read_size;
		//size_t cur_idx = 0;
		size_t cur_field = 0;
		char* cur_rbuf = _serial_read_buffer;
		while (true) {
			read_size = this->Read(read_file, cur_rbuf, SERIAL_READ_BUFFER_SIZE-std::distance(_serial_read_buffer, cur_rbuf));
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
		file_t& read_file = column.read_file_tls.local();
		if (unlikely(!read_file)) {
			this->Open(column.filename, &read_file, _direct_mode);
		}
		this->Seek(read_file, 0);

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
			read_size = this->Read(read_file, _serial_read_buffer, diff_off);
			if (read_size < 0) {
				assert(false);
				return false;
			} else if ((size_t)read_size != diff_off) {
				assert(false);
				return false;
			}
			const size_t processed_size = visit_column_process_chunk_sse(cur_field, end_field-1, _serial_read_buffer, read_size, f);
			if ((processed_size != (size_t)read_size) || (cur_field != end_field)) {
				assert(false);
				return false;
			}
			prev_off = off;
		}
		// Finish off !
		// We have an index at most every BUFFER_READ, so as SERIAL_READ_BUFFER_SIZE > BUFFER_READ, only one read is now
		// necessary !
		read_size = this->Read(read_file, _serial_read_buffer, SERIAL_READ_BUFFER_SIZE);
		visit_column_process_chunk_sse(cur_field, _nrows-1, _serial_read_buffer, read_size, f);

		return cur_field == _nrows;
	}

	template <typename F>
	bool visit_column2_sel(uint64_t const col_idx, F const& f, PVCore::PVSelBitField const& sel)
	{
		set_direct_mode(false);

		// Sequential version
		PVColumn& column = get_col(col_idx);
		file_t& read_file = column.read_file_tls.local();
		if (unlikely(!read_file)) {
			this->Open(column.filename, &read_file, _direct_mode);
		}
		this->Seek(read_file, 0);

		size_t rows_to_find = sel.get_number_of_selected_lines_in_range(0, _nrows);

		ssize_t read_size;
		size_t prev_off = 0;
		size_t cur_field = 0;
		typename index_table_t::column index_col = _indexes.get_col(col_idx);
		for (size_t i = 0; i < column.fields_indexed; i++) {
			offset_fields_t const& off_field = index_col.at(i);
			const size_t off = off_field.offset;

			// This field (`end_field') is not in the "previous" chunk, which
			// means that we are going to process fields in the interval [cur_field,end_field[ .
			const size_t end_field = off_field.field; 

			const size_t diff_off = off-prev_off;
			assert(diff_off <= SERIAL_READ_BUFFER_SIZE);
			// Check that something is selected in that chunk
			// is_empty_between is between [a,b[ (b is *not* included)
			if (!sel.is_empty_between(cur_field, end_field)) {
				this->Seek(read_file, prev_off);
				read_size = this->Read(read_file, _serial_read_buffer, diff_off);
				if (read_size < 0) {
					assert(false);
					return false;
				} else if ((size_t)read_size != diff_off) {
					assert(false);
					return false;
				}
				const size_t rows_found = visit_column_process_chunk_sel(cur_field, end_field-1, _serial_read_buffer, read_size, sel, f);
				//const size_t rows_found = sel.get_number_of_selected_lines_in_range(cur_field, end_field);
				assert(rows_found <= rows_to_find);
				assert(rows_found == sel.get_number_of_selected_lines_in_range(cur_field, end_field));
				if (rows_found == rows_to_find) {
					// That's the end of it!
					return true;
				}
				rows_to_find -= rows_found;
			}
			cur_field = end_field;
			prev_off = off;
		}
		// Finish off !
		// We have an index at most every BUFFER_READ, so as SERIAL_READ_BUFFER_SIZE > BUFFER_READ, only one read is now
		// necessary !
		this->Seek(read_file, prev_off);
		read_size = this->Read(read_file, _serial_read_buffer, SERIAL_READ_BUFFER_SIZE);
		visit_column_process_chunk_sel(cur_field, _nrows-1, _serial_read_buffer, read_size, sel, f);

		return true;
	}


	template <typename F>
	bool visit_column_tbb(uint64_t const col_idx, F const& f, tbb::task_group_context* ctxt = NULL)
	{
		set_direct_mode(false);

		// TBB version
		PVColumn& column = get_col(col_idx);
		file_t& read_file = column.read_file_tls.local();
		if (unlikely(!read_file)) {
			this->Open(column.filename, &read_file, _direct_mode);
		}
		this->Seek(read_file, 0);

		// Task context
		tbb::task_group_context my_ctxt;
		if (ctxt == NULL) {
			ctxt = &my_ctxt;
		}

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
					const ssize_t read_size = this->Read(read_file, chunk->buf, diff_off);
					if ((ssize_t) read_size != (ssize_t) diff_off) {
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
				}),

				*ctxt
		);

		if (ctxt->is_group_execution_cancelled()) {
			return false;
		}

		// Finish off !
		// We have an index at most every BUFFER_READ, so as SERIAL_READ_BUFFER_SIZE > BUFFER_READ, only one read is now
		// necessary !
		size_t read_size = this->Read(read_file, _serial_read_buffer, SERIAL_READ_BUFFER_SIZE);
		visit_column_process_chunk_sse(cur_field, _nrows-1, _serial_read_buffer, read_size, f);

		return cur_field == _nrows;
	}

	template <typename F>
	bool visit_column_tbb_sel(uint64_t const col_idx, F const& f, PVCore::PVSelBitField const& sel, tbb::task_group_context* ctxt = NULL)
	{
		set_direct_mode(false);

		// TBB version
		PVColumn& column = get_col(col_idx);
		file_t& read_file = column.read_file_tls.local();
		if (unlikely(!read_file)) {
			this->Open(column.filename, &read_file, _direct_mode);
		}
		this->Seek(read_file, 0);

		// Task context
		tbb::task_group_context my_ctxt;
		if (ctxt == NULL) {
			ctxt = &my_ctxt;
		}

		size_t prev_off = 0;
		size_t cur_field = 0;
		typename index_table_t::column index_col = _indexes.get_col(col_idx);
		size_t cur_idx = 0;
		ssize_t nrows_to_find = sel.get_number_of_selected_lines_in_range(0, _nrows);
		tbb::parallel_pipeline(_chunks.size(),
			tbb::make_filter<void, typename tbb_chunks_t::chunk_t*>(tbb::filter::serial_in_order,
				[&](tbb::flow_control& fc) -> typename tbb_chunks_t::chunk_t*
				{
					if (nrows_to_find <= 0) {
						// No more work to do!
						fc.stop();
						return nullptr;
					}

					size_t sel_lines_in_chunk = 0;
					size_t diff_off;
					size_t end_field;
					size_t off;
					while (cur_idx < column.fields_indexed) {
						offset_fields_t const& off_field = index_col.at(cur_idx);
						cur_idx++;
						off = off_field.offset;
						end_field = off_field.field;
						diff_off = off-prev_off;
						assert(diff_off <= SERIAL_READ_BUFFER_SIZE);
						sel_lines_in_chunk = sel.get_number_of_selected_lines_in_range(cur_field, end_field);
						if (sel_lines_in_chunk > 0) {
							break;
						}
						prev_off = off;
						cur_field = end_field;
					}
					if (sel_lines_in_chunk == 0) {
						// No more chunk with lines selected, that's the end.
						fc.stop();
						return nullptr;
					}

					nrows_to_find -= sel_lines_in_chunk;

					typename tbb_chunks_t::chunk_t* chunk = this->_chunks.get_chunk();
					this->Seek(read_file, prev_off);
					const ssize_t read_size = this->Read(read_file, chunk->buf, diff_off);
					if ((ssize_t) read_size != (ssize_t) diff_off) {
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
					this->visit_column_process_chunk_sel(c->start_field, c->end_field-1, &c->buf[0], c->size_data, sel, f);
					return c;
				}) &

			tbb::make_filter<typename tbb_chunks_t::chunk_t*, void>(tbb::filter::serial_out_of_order,
				[&](typename tbb_chunks_t::chunk_t* c)
				{
					this->_chunks.release_chunk(c);
				}),

				*ctxt
		);
		if (ctxt->is_group_execution_cancelled()) {
			return false;
		}

		// Finish off !
		// We have an index at most every BUFFER_READ, so as SERIAL_READ_BUFFER_SIZE > BUFFER_READ, only one read is now
		// necessary !
		this->Seek(read_file, prev_off);
		size_t read_size = this->Read(read_file, _serial_read_buffer, SERIAL_READ_BUFFER_SIZE);
		visit_column_process_chunk_sel(cur_field, _nrows-1, _serial_read_buffer, read_size, sel, f);

		return true;
	}

	bool get_unique_values_for_col(PVCol const c, unique_values_t& ret, tbb::task_group_context* ctxt = NULL);
	bool get_unique_values_for_col_with_sel(PVCol const c, unique_values_t& ret, PVCore::PVSelBitField const& sel, tbb::task_group_context* ctxt = NULL);

	bool count_by_with_sel(PVCol const col1, PVCol const col2, count_by_t& ret, PVCore::PVSelBitField const& sel, tbb::task_group_context* ctxt = nullptr);

	void clear_stats()
	{
		_stats_getindex = 0.0;
		_stats_read = 0.0;
		_stats_search = 0.0;
	}

	double get_stats_getindex() const { return (double)_stats_getindex/1000000.0; }
	double get_stats_read() const { return (double)_stats_read/1000000.0; }
	double get_stats_search() const { return (double)_stats_search/1000000.0; }

private:
	std::string get_disk_index_file() const;
	std::string get_disk_column_file(uint64_t col) const;
 
	/*! \brief Returns a string based on its index relative to a given buffer.
	 *
	 *  \param[in] col The column index.
	 *  \param[in] nb_fields The number of fields to skip.
	 *  \param[in] buffer The working buffer.
	 *  \param[out] buffer The returned string size.
	 *
	 *  \return A pointer to the buffer.
	 */
	char* next(uint64_t col, uint64_t nb_fields, char* buffer, size_t& size_ret);

	/*! \brief Close all the column files of the Nraw.
	 */
	void close_files();

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
		
		/*
		size_t fields_found_ref = 0;
		{
			for (size_t i = 0; i < size_buf; i++) {
				if (buf[i] == 0) {
					fields_found_ref++;
				}
			}
		}
		size_t found_field = 0;
		const size_t fields_asked = (field_end-field_start+1);*/
		size_t idx_start_field = 0;
		size_t i;
		const size_t size_buf_sse = (size_buf>>4)<<4;
		const __m128i sse_zero = _mm_setzero_si128();
		const __m128i sse_ff = _mm_set1_epi32(0xFFFFFFFF);
		for (i = 0; i < size_buf_sse; i += 16) {
			const __m128i sse_buf = _mm_load_si128((const __m128i*)&buf[i]);
			const __m128i sse_cmp = _mm_cmpeq_epi8(sse_buf, sse_zero);
			if (!_mm_testz_si128(sse_cmp, sse_ff)) {
				PVCore::PVByteVisitor::visit_bytes(sse_cmp,
					[&](size_t b)
					{
						//found_field++;
						f(field_start, &buf[idx_start_field], b-idx_start_field);
						idx_start_field = b+1;
						field_start++;
					},
					i
				);
				if (field_start > field_end) {
					//PVLOG_INFO("visit_column_process_chunk_sse: ask %llu fields, %llu found, %llu ref-found\n", (field_end-field_start+1), found_field, fields_found_ref);
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
				//found_field++;
				f(field_start, &buf[idx_start_field], i-idx_start_field);
				idx_start_field = i+1;
				field_start++;
				if (field_start > field_end) {
					break;
				}
			}
		}

		//PVLOG_INFO("visit_column_process_chunk_sse: ask %llu fields, %llu found, %llu ref-found\n", fields_asked, found_field, fields_found_ref);
		return idx_start_field;
	}

	template <typename F>
	size_t visit_column_process_chunk_sel(size_t const field_start, size_t const field_end, const char* buf, size_t size_buf, PVCore::PVSelBitField const& sel, F const& f)
	{
		// field_end is inclusive in the interface of this function, but not in visit_selected_lines (hence the '+1')
		const char* cur_buf = buf;
		const char* buf_end = buf+size_buf;
		size_t last_r = field_start;
		size_t nfound = 0;
		sel.visit_selected_lines([&](const PVRow r)
			{
				assert(r >= last_r);
				size_t slice_size;
				const char* slice = (const char*) PVCore::PVByteVisitor::get_nth_slice((const uint8_t*) cur_buf, (uintptr_t)buf_end-(uintptr_t)cur_buf, r-last_r, slice_size);
				f(r, (const char*) slice, slice_size);
				cur_buf = (const char*) (slice+slice_size+1);
				last_r = r+1;
				nfound++;
			},
			field_end+1,
			field_start);
		return nfound;
	}

	static constexpr size_t read_buffer_size() { return READ_BUFFER_SIZE+BUF_ALIGN; }


private:
	/**
	 * \class PVColumn
	 *
	 * \note
	 */
	struct PVColumn
	{
		typedef tbb::enumerable_thread_specific<file_t> tls_file_t;
		typedef tbb::enumerable_thread_specific<std::array<char, PVNrawDiskBackend::READ_BUFFER_SIZE+PVNrawDiskBackend::BUF_ALIGN>> tls_read_buffer_t;

	public:
		PVColumn():
			read_file_tls([](){ return 0; })
		{
			reset();
		}
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

			cache_index = INVALID;
			last_read_field = INVALID;
		}

	public:
		file_t write_file = 0;
		mutable tls_file_t read_file_tls;
		mutable tls_read_buffer_t read_buffer_tls;
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

		uint64_t cache_index;
		uint64_t last_read_field;
	};

	class PVCachePool: private file_policy_t
	{
	public:

		PVCachePool(PVRush::PVNrawDiskBackend& parent);
		~PVCachePool();

	public:
		/*! \brief Initialize the cache for a given couple row/column.
		 *
		 *  \param[in] field The field index.
		 *  \param[in] col The column index.
		 *
		 *  \note The cache buffer pointer is then located in "get_col(col).buffer_read_ptr"
		 *
		 *  \return The number of fields left in the cache.
		 */
		uint64_t get_cache(uint64_t field, uint64_t col);

		/*! \brief Returns the field index in the index map.
		 *
		 *  \param[in] col The column index.
		 *  \param[in] field The field index.
		 *
		 *  \return The field index or -1 in case of error.
		 */
		int64_t get_index(uint64_t col, uint64_t field) const;

	private:
		struct PVReadCache
		{
			char* buffer = nullptr;
			uint64_t column = INVALID;
			tbb::tick_count timestamp = tbb::tick_count::now();
			uint64_t first_field = INVALID;
			uint64_t last_field = INVALID; // This one is *not* included in this cache
		};

	private:
		PVReadCache _caches[NB_CACHE_BUFFERS];
		PVRush::PVNrawDiskBackend& _backend;
	};

private:
	/*! \brief Returns a reference to the PVColumn of the given index.
	 *
	 *  \param[in] col The column index.
	 *
	 *  \return A reference to the PVColumn of the given index.
	 */
	inline PVColumn& get_col(uint64_t col) { assert(col < _columns.size()); return _columns[col]; }
	inline const PVColumn& get_col(uint64_t col) const { assert(col < _columns.size()); return _columns[col]; }

	static bool merge_tls(unique_values_t& ret, tbb::enumerable_thread_specific<unique_values_t>& tbb_qset, tbb::task_group_context* ctxt = nullptr);

	bool merge_count_by_tls(count_by_t& ret, tbb::enumerable_thread_specific<count_by_t>& count_by_tls, tbb::task_group_context* ctxt = nullptr);

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

	size_t _nrows;
	char* _serial_read_buffer;

	tbb_chunks_t _chunks;

	mutable tbb::atomic<uint64_t> _stats_getindex;
	mutable tbb::atomic<uint64_t> _stats_read;
	mutable tbb::atomic<uint64_t> _stats_search;
};

}

#endif /* PVNRAWDISKBACKEND_H_ */
