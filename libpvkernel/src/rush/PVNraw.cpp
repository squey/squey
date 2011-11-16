/*
 * $Id: PVNraw.cpp 3226 2011-07-01 09:48:33Z aguinet $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 */

#include <pvkernel/rush/PVNraw.h>
#include <tbb/tick_count.h>
#include <iostream>

#define DEFAULT_LINE_SIZE 100

PVRush::PVNraw::PVNraw()
{
	_real_nrows = 0;
}

PVRush::PVNraw::~PVNraw()
{
	clear();
}

void PVRush::PVNraw::reserve(PVRow row, PVCol col)
{
	// Reserve memory for the table of the nraw
	clear_table();
	table.reserve(row);
	/*if (col > 0) {
		nraw_table::iterator it;
		for (it = table.begin(); it != table.end(); it++) {
			it->resize(col);
		}
	}*/
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
	tbb::tick_count start = tbb::tick_count::now();
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
	tbb::tick_count end = tbb::tick_count::now();
	PVLOG_INFO("(PVRush::PVNraw::create_trans_nraw) memory allocation took %0.4fs.\n", (end-start).seconds());

	start = end;
	nraw_table::const_iterator it;
	nraw_table_line::const_iterator it_col;
	PVCol c = 0;
	PVRow r = 0;
	/*for (it = table.begin(); it != table.end(); it++) {
		nraw_table_line const& cols = *it;
		for (it_col = cols.begin(); it_col != cols.end(); it_col++) {
			trans_table[c][r] = *it_col;
			c++;
		}
		c = 0;
		r++;
	}*/
	for (PVRow i = 0; i < nrows; i++) {
		for (PVCol j = 0; j < ncols; j++) {
			trans_table[j][i] = table[i][j];
		}
	}
	end = tbb::tick_count::now();
	PVLOG_INFO("(PVRush::PVNraw::create_trans_nraw) transposition took %0.4fs.\n", (end-start).seconds());
}

void PVRush::PVNraw::clear_table()
{
	nraw_table::const_iterator it;
	nraw_table_line::const_iterator it_l;
	for (it = table.begin(); it != table.end(); it++) {
		for (it_l = it->begin(); it_l != it->end(); it_l++) {
			static tbb::scalable_allocator<QChar> alloc;
			QString const& str = *it_l;
			alloc.deallocate((QChar*) str.constData(), str.size());
		}
	}
	table.clear();
	_real_nrows = 0;
}

void PVRush::PVNraw::copy(PVNraw &dst, PVNraw const& src)
{
	PVCol ncols = 0;
	PVCol nrows = src.table.size();
	if (nrows > 0) {
		ncols = src.table[0].size();
	}
	dst.reserve(nrows, ncols);
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

QString PVRush::PVNraw::nraw_line_to_csv(PVRow idx) const
{
	assert(idx < table.size());
	QString ret;
	PVRush::PVNraw::nraw_table_line const& line = table[idx];
	PVRush::PVNraw::nraw_table_line::const_iterator it,ite,itlast;
	ite = itlast = line.end();
	itlast--;
	for (it = line.begin(); it != ite; it++) {
		QString field = *it;
		if (field.indexOf(QChar(',')) >= 0 || field.indexOf(QChar('\r')) >= 0 || field.indexOf(QChar('\n')) >= 0) {
			field.replace(QChar('"'), QString("\\\""));
			ret += "\"" + field + "\"";
		}
		else 
			ret += field;
		if (it != itlast) {
			ret += QString(",");
		}
	}
	return ret;
}

void PVRush::PVNraw::fit_to_content()
{
	if (_real_nrows > PICVIZ_LINES_MAX) {
		_real_nrows = PICVIZ_LINES_MAX;
	}
	table.resize(_real_nrows);
	PVLOG_DEBUG("(PVNraw::fit_to_content) fit to content: size=%d.\n", table.size());
}

void PVRush::PVNraw::dump_csv()
{
	PVRush::PVNraw::nraw_table &nraw = get_table();
	PVRush::PVNraw::nraw_table::iterator it_nraw;
	PVRush::PVNraw::nraw_table_line::iterator it_nraw_line, it_nraw_line_end;
	for (it_nraw = nraw.begin(); it_nraw != nraw.end(); it_nraw++) {
		PVRush::PVNraw::nraw_table_line &l = *it_nraw;
		if (l.size() == 1) {
			QString &l_str = *(l.begin());
			std::cout << l_str.toUtf8().constData() << std::endl;
			continue;
		}
		it_nraw_line_end = l.end();
		it_nraw_line_end--;
		for (it_nraw_line = l.begin(); it_nraw_line != it_nraw_line_end; it_nraw_line++) {
			QString &field = *it_nraw_line;
			std::cout << "'" << field.toUtf8().constData() << "',";
		}
		QString &field = *it_nraw_line;
		std::cout << "'" << field.toUtf8().constData() << "'" << std::endl;
	}
}
