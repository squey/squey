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

PVRush::PVNraw::PVNraw()
{
	_real_nrows = 0;
	_chunks_todel = new list_chunks_t();
	_reallocated_buffers = new PVCore::buf_list_t();
}

PVRush::PVNraw::~PVNraw()
{
	clear();
	delete _chunks_todel;
	delete _reallocated_buffers;
}

void PVRush::PVNraw::reserve(PVRow row, PVCol col)
{
	clear_table();
	if (col == 0) {
		col = 1;
	}
#ifndef WIN32
	size_t max_alloc = (sysconf(_SC_AVPHYS_PAGES) * sysconf(_SC_PAGE_SIZE))/10;
	if (max_alloc > 0 && row*col*sizeof(PVCore::PVUnicodeString) > max_alloc) {
		row = max_alloc/(col*sizeof(PVCore::PVUnicodeString));
	}
#endif
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
	//tbb::tick_count start = tbb::tick_count::now();
	trans_table.clear();
	table.transpose_to(trans_table);
	//tbb::tick_count end = tbb::tick_count::now();
	//PVLOG_INFO("(PVNraw::create_trans_nraw) transposition took %0.4fs\n", (end-start).seconds());

	return true;
}

void PVRush::PVNraw::clear_table()
{
	_real_nrows = 0;
	table.clear();
	list_chunks_t::iterator it;
	if (_chunks_todel->size() > 0) {
		for (it = _chunks_todel->begin(); it != _chunks_todel->end(); it++) {
			(*it)->free();
		}
		_chunks_todel->clear();
	}

	{
		static tbb::tbb_allocator<char> alloc;
		PVCore::buf_list_t::const_iterator it;
		for (it = _reallocated_buffers->begin(); it != _reallocated_buffers->end(); it++) {
			alloc.deallocate(it->first, it->second);
		}
		_reallocated_buffers->clear();
	}
}

void PVRush::PVNraw::swap(PVNraw &dst, PVNraw& src)
{
	src.table.swap(dst.table);
	src.trans_table.swap(dst.trans_table);

	list_chunks_t* ltmp = dst._chunks_todel;
	dst._chunks_todel = src._chunks_todel;
	src._chunks_todel = ltmp;

	PVCore::buf_list_t* lbtmp = dst._reallocated_buffers;
	dst._reallocated_buffers = src._reallocated_buffers;
	src._reallocated_buffers = lbtmp;

	dst.format = src.format;
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
		else { 
			ret += field;
		}

		if (j != line.size()-1) {
			ret += QString(",");
		}
	}
	return ret;
}

QStringList PVRush::PVNraw::nraw_line_to_qstringlist(PVRow idx) const
{
	assert(idx < table.get_nrows());
	PVRush::PVNraw::nraw_table::const_line line = table[idx];
	QStringList stringlist;
	for (PVCol j = 0; j < line.size(); j++) {
		QString field = line[j].get_qstr();
		stringlist << field;
	}
	return stringlist;
}

void PVRush::PVNraw::fit_to_content()
{
	if (_real_nrows > PICVIZ_LINES_MAX) {
		_real_nrows = PICVIZ_LINES_MAX;
	}
	PVLOG_DEBUG("(PVNraw::fit_to_content) fit to content: size=%d.\n", table.get_nrows());
	table.resize_nrows(_real_nrows);
	PVLOG_DEBUG("(PVNraw::fit_to_content) fit to content done\n");
}

void PVRush::PVNraw::dump_csv()
{
	for (PVRow i = 0; i < table.get_nrows(); i++) {
		std::cout << qPrintable(nraw_line_to_csv(i)) << std::endl;
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
	_reallocated_buffers->splice(_reallocated_buffers->begin(), list);
}
