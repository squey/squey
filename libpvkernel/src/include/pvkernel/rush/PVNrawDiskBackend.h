/**
 * \file PVNrawDiskBackend.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef PVNRAWDISKBACKEND_H_
#define PVNRAWDISKBACKEND_H_

#include <sstream>
#include <string>
#include <fcntl.h>

#include <pvkernel/core/PVAllocators.h>

constexpr uint64_t BUF_ALIGN = 512;

namespace PVRush {

struct UnbufferedFilePolicy
{
	typedef int file_t;

	bool Open(std::string const& filename, file_t* file)
	{
		*file = open(filename.c_str(), O_RDWR | O_CREAT);
		return *file != -1;
	}

	inline int64_t Write(const char* content, uint64_t buf_size, file_t file)
	{
		return write(file, content, buf_size) != -1;
	}

	inline int64_t Read(file_t file, void* buffer,  uint64_t buf_size)
	{
		return read(file, buffer, buf_size);
	}

	inline int64_t Seek(file_t file, int64_t offset)
	{
		return lseek(file, offset, SEEK_CUR);
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

	inline bool Write(const char* content, uint64_t buf_size, file_t file)
	{
		return fwrite(content, buf_size, 1, file) > 0;
	}

	inline int64_t Read(file_t file, void* content, uint64_t buf_size)
	{
		return fread(content, 1, buf_size, file);
	}

	inline int64_t Seek(file_t file, int64_t offset)
	{
		return fseek(file, offset, SEEK_CUR);
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

struct RawFilePolicy : public UnbufferedFilePolicy
{
	bool Open(std::string const& filename, file_t* file)
	{
		*file = open(filename.c_str(), O_RDWR | O_CREAT | O_DIRECT);
		return *file != -1;
	}
};

template <typename FilePolicy = RawFilePolicy>
class PVNRawDiskBackend : public FilePolicy
{
public:
	typedef typename FilePolicy::file_t file_t;
public:
	PVNRawDiskBackend(std::string const& nraw_folder, uint64_t num_cols) : _nraw_folder(nraw_folder), _num_cols(num_cols)
	{
		_columns.reserve(_num_cols);
		for (int col = 0 ; col < _num_cols ; col++) {
			PVColumn column;

			// Open file
			std::stringstream filename;
			filename << nraw_folder << "/column_" << col;
			if(!this->Open(filename.str(), &column.file)) {
				PVLOG_ERROR("PVNRawDiskBackend: Error opening file %s (%s)\n", filename.str().c_str(), strerror(errno));
				return;
			}

			// Create buffer
			uint64_t buffer_size = _buffers_size_pattern[0];
			column.buffer = PVCore::PVAlignedAllocator<char, BUF_ALIGN>().allocate(buffer_size);
			column.buffer_ptr = column.buffer;
			column.buffer_end_ptr = column.buffer + buffer_size;

			_columns.push_back(std::move(column));
		}
	}

	void add(PVCol col_idx, const char* field, uint64_t field_size)
	{
		PVColumn& column = get_col(col_idx);
		uint64_t field_part2_size = 0;
		char* field_part2 = nullptr;

		// Fill the buffer with complete field
		if (column.buffer_ptr + field_size < column.buffer_end_ptr) {
			memcpy(column.buffer_ptr, field, field_size+1);
			column.buffer_ptr += field_size+1;
		}
		// Fill the buffer with splitted field
		else {
			uint64_t field_part1_size = column.buffer_end_ptr - column.buffer_ptr;
			memcpy(column.buffer_ptr, field, field_part1_size);
			field_part2 = const_cast<char*>(field) + field_part1_size; // ugly
			field_part2_size = field_size - field_part1_size +1;
			column.buffer_ptr += field_part1_size;
		}

		if (column.buffer_ptr == column.buffer_end_ptr) {
			// Buffer is full: write it to disk
			if(!this->Write(column.buffer, _buffers_size_pattern[column.buffers_size_idx], column.file)) {
				PVLOG_ERROR("PVNRawDiskBackend: Error writing column %d to disk (%s)\n", col_idx, strerror(errno));
				return;
			}

			uint64_t buffer_max_size = _buffers_size_pattern[column.buffers_size_idx];
			uint64_t max_size_idx = sizeof(_buffers_size_pattern)/sizeof(uint64_t)-1;
			if (column.buffers_size_idx < max_size_idx) {
				// Reallocate a bigger buffer and copy the end of splitted field
				PVCore::PVAlignedAllocator<char, BUF_ALIGN>().deallocate(column.buffer, buffer_max_size);
				uint64_t new_buffer_max_size = _buffers_size_pattern[++column.buffers_size_idx];
				column.buffer = PVCore::PVAlignedAllocator<char, BUF_ALIGN>().allocate(new_buffer_max_size);
				column.buffer_end_ptr = column.buffer + new_buffer_max_size;

				memcpy(column.buffer, field_part2, field_part2_size);
				column.buffer_ptr = column.buffer + field_part2_size;
			}
			else {
				// Recycle previously allocated buffer
				column.buffer_ptr = column.buffer;
			}
		}
	}

	void flush()
	{
		// WARNING: O_DIRECT option prevent to write a buffer of an arbitrary file (must be multiple of 512)

		/*for (uint64_t col_idx = 0 ; col_idx < _num_cols; col_idx++) {
		 	PVColumn& column = get_col(col_idx);
			uint64_t partial_buffer_size = column.buffer_end_ptr - column.buffer_ptr;
			if(!this->Write(column.buffer, partial_buffer_size,  column.file)) {
				PVLOG_ERROR("PVNRawDiskBackend: Error writing column %d to disk (%s)\n", col_idx, strerror(errno));
				return;
			}
		}*/
	}

	char* at(PVRow row, PVCol col) const
	{

	}

	~PVNRawDiskBackend()
	{
		// Close files and deallocate all buffers
		for (uint64_t col_idx = 0 ; col_idx < _num_cols; col_idx++) {
			PVColumn& column = get_col(col_idx);
			this->Close(column.file);
			PVCore::PVAlignedAllocator<char, BUF_ALIGN>().deallocate(column.buffer, _buffers_size_pattern[column.buffers_size_idx]);
		}
	}

private:
	struct PVColumn
	{
		file_t file = 0;
		char* buffer = nullptr;
		char* buffer_ptr = nullptr;
		char* buffer_end_ptr = nullptr;
		uint64_t buffers_size_idx = 0;
	};

private:
	inline PVColumn & get_col(uint64_t col) { return _columns[col]; }

private:
	const std::string& _nraw_folder;
	uint64_t _num_cols;

	std::vector<PVColumn> _columns;

	uint64_t _buffers_size_pattern[8] = { // too bad we need to specify array size...
		128*1024,
		256*1024,
		512*1024,
		1*1024*1024,
		2*1024*1024,
		4*1024*1024,
		8*1024*1024,
		16*1024*1024
	};
};

}

#endif /* PVNRAWDISKBACKEND_H_ */
