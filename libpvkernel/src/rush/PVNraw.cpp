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
	_chunks_todel = new list_chunks_t();
}

PVRush::PVNraw::~PVNraw()
{
	clear();
	delete _chunks_todel;
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
	free_trans_nraw();
	clear_table();
}

bool PVRush::PVNraw::create_trans_nraw()
{
	// Create a transposition of the nraw
	tbb::tick_count start = tbb::tick_count::now();
	trans_table.clear();
	table.transpose_to(trans_table);
	tbb::tick_count end = tbb::tick_count::now();
	PVLOG_INFO("(PVNraw::create_trans_nraw) transposition took %0.4fs\n", (end-start).seconds());

	return true;
}

void PVRush::PVNraw::clear_table()
{
	_real_nrows = 0;
	table.clear();
	list_chunks_t::iterator it;
	for (it = _chunks_todel->begin(); it != _chunks_todel->end(); it++) {
		(*it)->free();
	}
	_chunks_todel->clear();

	{
		static tbb::scalable_allocator<char> alloc;
		PVCore::buf_list_t::const_iterator it;
		for (it = _reallocated_buffers.begin(); it != _reallocated_buffers.end(); it++) {
			alloc.deallocate(it->first, it->second);
		}
		_reallocated_buffers.clear();
	}
}

void PVRush::PVNraw::swap(PVNraw &dst, PVNraw& src)
{
	src.table.swap(dst.table);
	src.trans_table.swap(dst.trans_table);

	list_chunks_t* ltmp = dst._chunks_todel;
	dst._chunks_todel = src._chunks_todel;
	src._chunks_todel = ltmp;

	PVCore::buf_list_t lbtmp = dst._reallocated_buffers;
	dst._reallocated_buffers = src._reallocated_buffers;
	src._reallocated_buffers = lbtmp;

	//dst.format.swap(src.format);
}

QString PVRush::PVNraw::nraw_line_to_csv(PVRow idx) const
{
	assert(idx < table.get_nrows());
	QString ret;
	PVRush::PVNraw::nraw_table::const_line line = table[idx];
	for (PVCol j = 0; j < line.size(); j++) {
		QString field = line[j].get_qstr();
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

void PVRush::PVNraw::take_realloc_buffers(PVCore::buf_list_t& list)
{
	std::copy(list.begin(), list.end(), _reallocated_buffers.end());
	list.clear();
}
