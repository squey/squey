/*
 * $Id: PVNraw.cpp 3226 2011-07-01 09:48:33Z aguinet $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 */

#include <pvrush/PVNraw.h>

#define DEFAULT_LINE_SIZE 100

PVRush::PVNraw::PVNraw() :
	_buf_strings(NULL)
{
}

PVRush::PVNraw::~PVNraw()
{
	clear_table();
	//delete_buffers();
}

QString PVRush::PVNraw::get_value(PVRow row, PVCol col)
{
	return table.at(row)[col];
}

void PVRush::PVNraw::allocate_buf(size_t nchars)
{
	// Reserve the "big buffer" that will hold our strings
	static tbb::tbb_allocator<QChar> alloc;
	_buf_strings = alloc.allocate(nchars);
	_len_buf = _rem_len_buf = nchars;
	_cur_buf = _buf_strings;
	_buf_todel.push_back(std::pair<QChar*, size_t>(_buf_strings, _len_buf));
}

void PVRush::PVNraw::reserve(PVRow row)
{
	// Reserve memory for the table of the nraw
	clear_table();
	//delete_buffers();
	_reserved_lines = row;
	table.reserve(row);
	//allocate_buf(row*DEFAULT_LINE_SIZE);
}

void PVRush::PVNraw::free_trans_nraw()
{
	// Free the memory taken by the transposed table
	trans_table.clear();
}

void PVRush::PVNraw::clear()
{
	trans_table.clear();
	clear_table();
	//delete_buffers();
}

void PVRush::PVNraw::create_trans_nraw()
{
	// Create a transposition of the nraw
	PVCol ncols = table[0].size();
	PVRow nrows = table.size();
	trans_table.resize(ncols);

	{
		nraw_trans_table::iterator it;
		for (it = trans_table.begin(); it != trans_table.end(); it++) {
			(*it).resize(nrows);
		}
	}

	nraw_table::const_iterator it;
	nraw_table_line::const_iterator it_col;
	PVCol c = 0;
	PVRow r = 0;
	for (it = table.begin(); it != table.end(); it++) {
		nraw_table_line const& cols = *it;
		for (it_col = cols.begin(); it_col != cols.end(); it_col++) {
			trans_table[c][r] = *it_col;
			c++;
		}
		c = 0;
		r++;
	}
}

void PVRush::PVNraw::delete_buffers()
{
	static tbb::tbb_allocator<QChar> alloc;
	std::list<std::pair<QChar*, size_t> >::iterator it;
	for (it = _buf_todel.begin(); it != _buf_todel.end(); it++) {
		alloc.deallocate(it->first, it->second);
	}
	_buf_todel.clear();
}

void PVRush::PVNraw::clear_table()
{
	nraw_table::const_iterator it;
	nraw_table_line::const_iterator it_l;
	for (it = table.begin(); it != table.end(); it++) {
		for (it_l = it->begin(); it_l != it->end(); it_l++) {
			static tbb::tbb_allocator<QChar> alloc;
			QString const& str = *it_l;
			alloc.deallocate((QChar*) str.constData(), str.size());
		}
	}
	table.clear();
}

void PVRush::PVNraw::copy(PVNraw &dst, PVNraw const& src)
{
	dst.reserve(src.table.size());
	nraw_table::const_iterator it;
	nraw_table_line::const_iterator it_l;
	for (it = src.table.begin(); it != src.table.end(); it++) {
		nraw_table_line &line = dst.add_row(it->size());
		size_t i = 0;
		for (it_l = it->begin(); it_l != it->end(); it_l++) {
			QString const& str = *it_l;
			dst.set_field(line, i, str.constData(), str.size());
			i++;
		}
	}

	if (src.trans_table.size() > 0) {
		dst.create_trans_nraw();
	}
	dst.format = src.format;
}

void PVRush::PVNraw::move(PVNraw &dst, PVNraw& src)
{
	dst.table = src.table;
	dst.trans_table = src.trans_table;
	dst.format = src.format;

	src.table.clear();
	src.trans_table.clear();
}
