/**
 * \file PVNrawDiskBackend.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvbase/qhashes.h>
#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/PVDirectory.h>
#include <pvkernel/rush/PVNrawDiskBackend.h>

#include <tbb/enumerable_thread_specific.h>

static const std::string INDEX_FILENAME = std::string("nraw.idx");

// class tbb_chunks_t
PVRush::PVNrawDiskBackend::tbb_chunks_t::tbb_chunks_t(size_t n):
			_n(n)
{
	_chunks = PVCore::PVAlignedAllocator<chunk_t, 16>().allocate(n);
	_queue.set_capacity(n);
	for (size_t i = 0; i < n; i++) {
		_queue.try_push(&_chunks[i]);
	}
}

PVRush::PVNrawDiskBackend::tbb_chunks_t::~tbb_chunks_t()
{
	PVCore::PVAlignedAllocator<chunk_t, 16>().deallocate(_chunks, _n);
}


// class PVNrawDiskBackend
PVRush::PVNrawDiskBackend::PVNrawDiskBackend():
		_cache_pool(*this),
		_nrows(0),
		_serial_read_buffer(nullptr),
		_chunks(48)
{
}

PVRush::PVNrawDiskBackend::~PVNrawDiskBackend()
{
	if (_serial_read_buffer) {
		PVCore::PVAlignedAllocator<char, BUF_ALIGN>().deallocate(_serial_read_buffer, SERIAL_READ_BUFFER_SIZE);
	}
	close_files();
}

void PVRush::PVNrawDiskBackend::close_files()
{
	// Close files
	for (uint64_t col_idx = 0 ; col_idx < get_number_cols(); col_idx++) {
		PVColumn& column = get_col(col_idx);
		this->Close(column.file);
	}
}

void PVRush::PVNrawDiskBackend::init(const char* nraw_folder, const uint64_t num_cols)
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

void PVRush::PVNrawDiskBackend::clear_and_remove()
{
	clear();
	close_files();
	PVCore::PVDirectory::remove_rec(QString::fromLocal8Bit(_nraw_folder.c_str()));
}

uint64_t PVRush::PVNrawDiskBackend::add(PVCol col_idx, const char* field, const uint64_t field_size)
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
		column.buffer_write_ptr += written_field_size;
		assert(column.buffer_write_ptr <= column.buffer_write_end_ptr);
	}
	// Fill the buffer_write with splitted field
	else {
		assert((uintptr_t)column.buffer_write_end_ptr >= (uintptr_t)column.buffer_write_ptr);
		uint64_t field_part1_size = column.buffer_write_end_ptr - column.buffer_write_ptr;
		memcpy(column.buffer_write_ptr, field, field_part1_size);
		field_part2 = (char *)(field + field_part1_size);
		field_part2_size = written_field_size - field_part1_size;
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
			memcpy(column.buffer_write, field_part2, field_part2_size-1); // '-1' because there is no trailing '\0' in field_part2
			column.buffer_write[field_part2_size-1] = 0;
			column.buffer_write_ptr = column.buffer_write + field_part2_size;
			assert(column.buffer_write_ptr <= column.buffer_write_end_ptr);
		}
	}
	_nrows = std::max(_nrows, column.fields_nb);

	return write_size;
}

void PVRush::PVNrawDiskBackend::flush()
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
		//column.reset();
	}

	store_index_to_disk();
}

uint64_t PVRush::PVNrawDiskBackend::search_in_column(uint64_t col_idx, std::string const& field)
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

void PVRush::PVNrawDiskBackend::set_direct_mode(bool direct)
{
	if (_direct_mode == direct) {
		return;
	}

	for (size_t col = 0 ; col < get_number_cols() ; col++) {
		PVColumn& column = get_col(col);
		uint64_t pos = this->Tell(column.file);
		this->Close(column.file);
		this->Open(column.filename, &column.file, direct);
		this->Seek(column.file, pos);
	}
	_direct_mode = direct;
}

void PVRush::PVNrawDiskBackend::store_index_to_disk()
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

void PVRush::PVNrawDiskBackend::load_index_from_disk()
{
	// TODO: wrong size computation!
	set_direct_mode(false);
	file_t file;
	this->Open(get_disk_index_file(), &file, false);
	if (this->Read(file, (size_t*) &_nrows, sizeof(size_t)) != sizeof(size_t)) {
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

void PVRush::PVNrawDiskBackend::clear()
{
	unlink(get_disk_index_file().c_str());
	for (uint64_t c = 0 ; c < get_number_cols() ; c++) {
		PVColumn& nraw_c = _columns[c];
		this->Truncate(nraw_c.file, 0);

		nraw_c.buffer_write_ptr = nraw_c.buffer_write;
		nraw_c.field_length = 0; // Or any value grater than 0 to specify a fixed field length;
	}

	//_next_indexes_nrows = _index_fields_size_pattern[0];
	//_indexes.resize_nrows(_next_indexes_nrows);
	_fields_size_idx = 0;
	_next_indexes_nrows += _index_fields_size_pattern[++_fields_size_idx];
	_nrows = 0;
}

void PVRush::PVNrawDiskBackend::print_stats()
{
	PVLOG_INFO("search: %d msec\n", this->_search_interval.seconds()*1000);
	PVLOG_INFO("resize: %d msec\n", this->_matrix_resize_interval.seconds()*1000);
}

void PVRush::PVNrawDiskBackend::print_indexes()
{
	for (uint64_t r=0; r < _indexes_nrows; r++) {
		for (uint64_t c=0; c < get_number_cols(); c++) {
			PVLOG_INFO("index[%d][%d] = %d (offset), %d (field)\n", r, c, _indexes.at(r, c).offset, _indexes.at(r, c).field);
		}
	}
}

char* PVRush::PVNrawDiskBackend::next(uint64_t col, uint64_t nb_fields, char* buffer, size_t& size_ret)
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
		uint64_t size_to_read = READ_BUFFER_SIZE - (buffer - column.buffer_read_ptr);
		// TODO: vectorize this!
		/*for (uint64_t i = 0; i < nb_fields; i++) {
			end_field_ptr = (char*) memchr(buffer_ptr, '\0', size_to_read);
			assert(end_field_ptr && ((uintptr_t)end_field_ptr - (uintptr_t)column.buffer_read <= (size_t)READ_BUFFER_SIZE));
			uint64_t field_length = (end_field_ptr - buffer_ptr)+1;
			size_to_read -= field_length;
			buffer_ptr += field_length;
		}*/
		//size_ret = strnlen(buffer_ptr, size_to_read);
		PVCore::PVByteVisitor::visit_nth_slice((const uint8_t*) buffer, size_to_read, nb_fields,  [&](const uint8_t* found_buf, size_t sbuf) { buffer_ptr = (char*) found_buf; size_ret = sbuf; });
	}

	column.buffer_read_ptr = buffer_ptr;

	tbb::tick_count t2 = tbb::tick_count::now();
	_search_interval += (t2-t1);
	return buffer_ptr;
}

std::string PVRush::PVNrawDiskBackend::get_disk_index_file() const
{
	std::stringstream filename;
	filename << _nraw_folder << "/" << INDEX_FILENAME;
	return std::move(filename.str());
}

std::string PVRush::PVNrawDiskBackend::get_disk_column_file(uint64_t col) const
{
	assert(col < get_number_cols());
	std::stringstream filename;
	filename << _nraw_folder << "/column_" << col;
	return std::move(filename.str());
}

// class PVCachePool
PVRush::PVNrawDiskBackend::PVCachePool::PVCachePool(PVRush::PVNrawDiskBackend& parent) : _parent(parent)
{
	for (uint64_t cache_idx = 0; cache_idx < NB_CACHE_BUFFERS; cache_idx++) {
		_caches[cache_idx].buffer = PVCore::PVAlignedAllocator<char, BUF_ALIGN>().allocate(READ_BUFFER_SIZE);
	}
}

uint64_t PVRush::PVNrawDiskBackend::PVCachePool::get_cache(uint64_t field, uint64_t col)
{
	bool cache_miss = true;

	PVColumn& column = _parent.get_col(col);
	uint64_t cache_idx = column.cache_index;

	// Is there a cache for this column?

	// Yes
	if (cache_idx != INVALID) {
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
		if (_caches[cache_idx].column != INVALID) {
			_parent.get_col(_caches[cache_idx].column).cache_index = INVALID;  // Tell the column that we are stealing its cache...
		}

		column.cache_index = cache_idx;
		_caches[cache_idx].column = col;
	}

	PVReadCache& cache = _caches[cache_idx];

	// Fetch data from disk
	uint64_t nb_fields_left = 0;
	if (cache_miss) {
		char* buffer_ptr = cache.buffer;
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
		nb_fields_left = field - cache.first_field;
		column.buffer_read = buffer_ptr;
		column.buffer_read_ptr = buffer_ptr;
	}
	else {
		if (field > column.last_read_field) {
			nb_fields_left = field - column.last_read_field;
		}
		else {
			nb_fields_left = field - cache.first_field;
			column.buffer_read_ptr = column.buffer_read;
		}
	}

	// Update cache timestamp
	cache.timestamp = tbb::tick_count::now();
	column.last_read_field = field;

	return nb_fields_left;
}

PVRush::PVNrawDiskBackend::PVCachePool::~PVCachePool()
{
	for (uint64_t cache_idx = 0; cache_idx < NB_CACHE_BUFFERS; cache_idx++) {
		PVCore::PVAlignedAllocator<char, BUF_ALIGN>().deallocate(_caches[cache_idx].buffer, READ_BUFFER_SIZE);
	}
}

int64_t PVRush::PVNrawDiskBackend::PVCachePool::get_index(uint64_t col, uint64_t field)
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
	const size_t findexed = _parent.get_col(col).fields_indexed;
	if (findexed == 0) {
		return -1;
	}

	for (int64_t index = findexed-1; index >= 0; index--) {
		if (indexes.at(index, col).field <= field) {
			return index;
		}
	}
	return -1;
}

bool PVRush::PVNrawDiskBackend::get_unique_values_for_col(PVCol const c, unique_values_t& ret, tbb::task_group_context* ctxt)
{
	const size_t nreserve = std::sqrt(_nrows);
	tbb::enumerable_thread_specific<unique_values_t> tbb_qset([nreserve]{ unique_values_t ret; ret.reserve(nreserve); return ret; }); 
	bool vret = visit_column_tbb(c, [&tbb_qset](size_t, const char* buf, size_t n)
			{   
				std::string_tbb new_s(buf, n); 
				tbb_qset.local().insert(new_s);
			}, ctxt);
	if (!vret) {
		return false;
	}
	typename decltype(tbb_qset)::iterator it_tls = tbb_qset.begin();
	unique_values_t& final = *it_tls;
	it_tls++;
	for (; it_tls != tbb_qset.end(); it_tls++) {
		final.unite(*it_tls);
	}   
	ret = final;
	return true;
}

bool PVRush::PVNrawDiskBackend::get_unique_values_for_col_with_sel(PVCol const c, unique_values_t& ret, PVCore::PVSelBitField const& sel, tbb::task_group_context* ctxt)
{
	// AG: TODO: factorize w/ the previous function!
	const size_t nreserve = std::sqrt(_nrows);
	tbb::enumerable_thread_specific<unique_values_t> tbb_qset([nreserve]{ unique_values_t ret; ret.reserve(nreserve); return ret; }); 
	bool vret = visit_column_tbb_sel(c, [&tbb_qset](size_t, const char* buf, size_t n)
			{   
				std::string_tbb new_s(buf, n); 
				tbb_qset.local().insert(new_s);
			}, sel, ctxt); 
	if (!vret) {
		return false;
	}
	typename decltype(tbb_qset)::iterator it_tls = tbb_qset.begin();
	unique_values_t& final = *it_tls;
	it_tls++;
	for (; it_tls != tbb_qset.end(); it_tls++) {
		final.unite(*it_tls);
	}   
	ret = final;
	return true;
}