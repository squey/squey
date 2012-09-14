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
#include <pvkernel/core/PVMatrix.h>

constexpr uint64_t BUF_ALIGN = 512;
constexpr uint64_t READ_BUFFER_SIZE = 256*1024;
constexpr uint64_t FIELDS_PER_INDEX = 8192;

namespace PVRush {

/*
 *
 * File policy classes
 *
 *
 */

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
		return write(file, content, buf_size);
	}

	inline int64_t Read(file_t file, void* buffer,  uint64_t buf_size)
	{
		return read(file, buffer, buf_size);
	}

	inline int64_t ReadAt(file_t file, uint64_t offset, void* buffer, uint64_t buf_size)
	{
		lseek(file, 0, SEEK_SET);
		return read(file, buffer, buf_size);
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
		fseek(file, 0, SEEK_SET);
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

struct RawFilePolicy : public UnbufferedFilePolicy
{
	bool Open(std::string const& filename, file_t* file)
	{
		*file = open(filename.c_str(), O_RDWR | O_CREAT | O_DIRECT);
		return *file != -1;
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
	typedef PVCore::PVMatrix<uint64_t, PVCol, PVRow> index_table_t;

public:
	PVNRawDiskBackend(std::string const& nraw_folder, uint64_t num_cols) :
		_nraw_folder(nraw_folder),
		_num_cols(num_cols),
		_indexes(10, num_cols)
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
			column.buffer = PVCore::PVAlignedAllocator<char, BUF_ALIGN>().allocate(buffer_size);
			column.buffer_ptr = column.buffer;
			column.buffer_end_ptr = column.buffer + buffer_size;
			column.field_length = 9; // Or any value grater than 0 to specify a fixed field length;
		}

		_read_buffer = PVCore::PVAlignedAllocator<char, BUF_ALIGN>().allocate(READ_BUFFER_SIZE);
	}

	uint64_t add(PVCol col_idx, const char* field, uint64_t field_size)
	{
		PVColumn& column = get_col(col_idx);
		uint64_t field_part2_size = 0;
		char* field_part2 = nullptr;
		uint64_t write_size = 0;

		// Index field
		if (++column.fields_ignored == FIELDS_PER_INDEX) {
			uint64_t field_offset_in_file = this->Tell(column.file) + (column.buffer_ptr - column.buffer);
			_indexes.resize(++column.fields_indexed+1, _num_cols);
			_indexes.set_value(column.fields_indexed, col_idx, field_offset_in_file);
			std::cout << "_indexes[" << col_idx << "][" << column.fields_indexed-1 << "]=" << field_offset_in_file << std::endl;
			column.fields_ignored = 0;
		}

		// Fill the buffer with complete field
		field_size += column.end_char();
		if (column.buffer_ptr + field_size <= column.buffer_end_ptr) {
			memcpy(column.buffer_ptr, field, field_size);
			column.buffer_ptr += field_size;
		}
		// Fill the buffer with splitted field
		else {
			uint64_t field_part1_size = column.buffer_end_ptr - column.buffer_ptr;
			memcpy(column.buffer_ptr, field, field_part1_size);
			field_part2 = (char *)(field + field_part1_size);
			field_part2_size = field_size - field_part1_size;
			column.buffer_ptr += field_part1_size;
		}

		// Write buffer to disk
		if (column.buffer_ptr == column.buffer_end_ptr) {
			write_size = this->Write(column.buffer, _buffers_size_pattern[column.buffers_size_idx], column.file);
			if(write_size <= 0) {
				PVLOG_ERROR("PVNRawDiskBackend: Error writing column %d to disk (%s)\n", col_idx, strerror(errno));
				return 0;
			}

			uint64_t buffer_max_size = _buffers_size_pattern[column.buffers_size_idx];

			// Reallocate a bigger buffer and copy the end of splitted field
			if (column.buffers_size_idx < _max_size_idx) {
				uint64_t new_buffer_max_size = _buffers_size_pattern[++column.buffers_size_idx];
				PVCore::PVAlignedAllocator<char, BUF_ALIGN>().deallocate(column.buffer, buffer_max_size);
				column.buffer = PVCore::PVAlignedAllocator<char, BUF_ALIGN>().allocate(new_buffer_max_size);
				column.buffer_end_ptr = column.buffer + new_buffer_max_size;

				memcpy(column.buffer, field_part2, field_part2_size);
				column.buffer_ptr = column.buffer + field_part2_size;
			}
			// Recycle previously allocated buffer
			else {
				column.buffer_ptr = column.buffer;
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

		// Load buffer
		uint64_t field_index = field / FIELDS_PER_INDEX;
		uint64_t disk_offset = _indexes.at(field_index, col);
		this->ReadAt(column.file, disk_offset, _read_buffer, READ_BUFFER_SIZE);

		// Extract field
		char* buffer_ptr = _read_buffer;
		uint64_t num_field_in_index = field_index * FIELDS_PER_INDEX;
		if (column.field_length > 0) {
			return buffer_ptr + ((field -num_field_in_index) * column.field_length);
		}
		else {
			char* end_field_ptr = nullptr;
			uint64_t size_to_read = READ_BUFFER_SIZE;
			for (num_field_in_index; num_field_in_index < field; num_field_in_index++) {
				end_field_ptr = (char*) memchr(buffer_ptr, '\0', size_to_read);
				uint64_t field_length = (end_field_ptr - buffer_ptr);
				size_to_read -= field_length;
				buffer_ptr += (field_length+1);
			}
		}

		return buffer_ptr;
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

	~PVNRawDiskBackend()
	{
		// Close files and deallocate all buffers
		for (uint64_t col_idx = 0 ; col_idx < _num_cols; col_idx++) {
			PVColumn& column = get_col(col_idx);
			this->Close(column.file);
			PVCore::PVAlignedAllocator<char, BUF_ALIGN>().deallocate(column.buffer, _buffers_size_pattern[column.buffers_size_idx]);
		}
		PVCore::PVAlignedAllocator<char, BUF_ALIGN>().deallocate(_read_buffer, READ_BUFFER_SIZE);
	}

private:
	struct PVColumn
	{
	public:
		bool end_char() { return field_length == 0; }

	public:
		file_t file = 0;

		char* buffer = nullptr;
		char* buffer_ptr = nullptr;
		char* buffer_end_ptr = nullptr;
		uint64_t buffers_size_idx = 0;

		uint64_t field_length = 0;
		uint64_t fields_ignored = 0;
		uint64_t fields_indexed = 0;
	};

private:
	inline PVColumn& get_col(uint64_t col) { return _columns[col]; }

private:
	const std::string& _nraw_folder;
	uint64_t _num_cols;

	std::vector<PVColumn> _columns;
	index_table_t _indexes;

	char* _read_buffer = nullptr;

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
	uint64_t _max_size_idx = sizeof(_buffers_size_pattern)/sizeof(uint64_t)-1;
};

}

#endif /* PVNRAWDISKBACKEND_H_ */
