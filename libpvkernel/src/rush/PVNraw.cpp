/**
 * \file PVNraw.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/rush/PVNraw.h>
#include <unistd.h>
#include <tbb/tick_count.h>
#include <iostream>
#include <stdio.h>

#define DEFAULT_LINE_SIZE 100
#define MAX_SIZE_RESERVE (size_t(1024*1024*1024u)) // 1GB

PVRush::PVNraw::PVNraw():
	_tmp_conv_buf(nullptr)
{
	_real_nrows = 0;

	UErrorCode status = U_ZERO_ERROR;
	_ucnv = ucnv_open("UTF8", &status);
}

PVRush::PVNraw::~PVNraw()
{
	ucnv_close(_ucnv);
	clear();
}

void PVRush::PVNraw::reserve(PVRow const /*nrows*/, PVCol const ncols)
{
	_backend.init("/srv/data-r0/nraw", ncols);
}

void PVRush::PVNraw::clear()
{
	if (_tmp_conv_buf) {
		tbb::scalable_allocator<char>().deallocate(_tmp_conv_buf, _tmp_conv_buf_size);
	}
	_real_nrows = 0;
}

void PVRush::PVNraw::swap(PVNraw &dst, PVNraw& src)
{
	dst.format = src.format;
}

void PVRush::PVNraw::dump_csv()
{
	for (PVRow i = 0; i < get_number_rows(); i++) {
		std::cout << qPrintable(nraw_line_to_csv(i)) << std::endl;
	}
}

void PVRush::PVNraw::reserve_tmp_buf(size_t n)
{
	if (_tmp_conv_buf) {
		if (_tmp_conv_buf_size < n) {
			tbb::scalable_allocator<char>().deallocate(_tmp_conv_buf, _tmp_conv_buf_size);
			_tmp_conv_buf = tbb::scalable_allocator<char>().allocate(n);
			_tmp_conv_buf_size = n;
		}
	}
	else {
		_tmp_conv_buf = tbb::scalable_allocator<char>().allocate(n);
		_tmp_conv_buf_size = n;
	}
}

bool PVRush::PVNraw::add_chunk_utf16(PVCore::PVChunk const& chunk)
{
	// Write all elements of the chunk in the final nraw
	PVCore::list_elts const& elts = chunk.c_elements();	
	PVCore::list_elts::const_iterator it_elt;

	UErrorCode err = U_ZERO_ERROR;
	for (it_elt = elts.begin(); it_elt != elts.end(); it_elt++) {
		PVCore::PVElement& e = *(*it_elt);
		if (!e.valid())
			continue;
		PVCore::list_fields const& fields = e.c_fields();
		if (fields.size() == 0)
			continue;

		_real_nrows++;
		PVCol col = 0;
		PVCore::list_fields::const_iterator it_field;
		for (it_field = fields.begin(); it_field != fields.end(); it_field++) {
			// Convert to UTF8
			// TODO: make the whole process in utf8.. !
			PVCore::PVField const& field = *it_field;
			reserve_tmp_buf(field.size());
			const size_t size_utf8 = ucnv_fromUChars(_ucnv, _tmp_conv_buf, _tmp_conv_buf_size, (const UChar*) field.begin(), field.size()/sizeof(UChar), &err);
			if (!U_SUCCESS(err)) {
				PVLOG_WARN("Unable to convert field %d to UTF8! Field is ignored..\n", col);
				continue;
			}

			_backend.add(col, _tmp_conv_buf, size_utf8);
			col++;
		}
	}

	return true;
}

void PVRush::PVNraw::fit_to_content()
{
	_backend.flush();
	if (_real_nrows > PICVIZ_LINES_MAX) {
		_real_nrows = PICVIZ_LINES_MAX;
	}
}

QString PVRush::PVNraw::get_value(PVRow row, PVCol col) const
{
	assert(row < get_number_rows());
	assert(col < get_number_cols());
	size_t size = 0;
	const char* buf = _backend.at(row, col, size);
	return QString::fromUtf8(buf, size);
}

QString PVRush::PVNraw::nraw_line_to_csv(PVRow idx) const
{
	return QString();
}
