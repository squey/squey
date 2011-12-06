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
	clear_table();
	if (col == 0) {
		col = 1;
	}
	table.resize(row, col);
}

void PVRush::PVNraw::free_trans_nraw()
{
	// Free the memory taken by the transposed table
	trans_table.free();
}

void PVRush::PVNraw::clear()
{
	trans_table.clear();
	clear_table();
}

bool PVRush::PVNraw::create_trans_nraw()
{
	// Create a transposition of the nraw
	trans_table.clear();
	table.transpose_to(trans_table);
	return true;
}

void PVRush::PVNraw::clear_table()
{
	_real_nrows = 0;
	for (PVRow r = 0; r < table.get_nrows(); r++) {
		PVCore::PVField* first_field = *(table.get_row_ptr(r));
		first_field->elt_parent()->chunk_parent()->free();
	}
	table.clear();
}

void PVRush::PVNraw::copy(PVNraw &dst, PVNraw const& src)
{
	// TODO!
	if (src.trans_table.get_nrows() > 0) {
		dst.create_trans_nraw();
	}
	dst.format = src.format;
}

void PVRush::PVNraw::move(PVNraw &dst, PVNraw& src)
{
	// TODO!
	src.table.clear();
	src.trans_table.clear();
}

QString PVRush::PVNraw::nraw_line_to_csv(PVRow idx) const
{
	assert(idx < table.get_nrows());
	QString ret;
	PVRush::PVNraw::nraw_table::const_line line = table[idx];
	for (PVCol j = 0; j < line.size(); j++) {
		QString field = line[j]->get_qstr();
		if (field.indexOf(QChar(',')) >= 0 || field.indexOf(QChar('\r')) >= 0 || field.indexOf(QChar('\n')) >= 0) {
			field.replace(QChar('"'), QString("\\\""));
			ret += "\"" + field + "\"";
		}
		else 
			ret += field;
		if (j != line.size()-1) {
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
	table.resize_nrows(_real_nrows);
	PVLOG_DEBUG("(PVNraw::fit_to_content) fit to content: size=%d.\n", table.get_nrows());
}

void PVRush::PVNraw::dump_csv()
{
	for (PVRow i = 0; i < table.get_nrows(); i++) {
		nraw_line_to_csv(i);
	}
#if 0
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
#endif
}
