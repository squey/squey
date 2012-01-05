/*
 * $Id: PVNraw.cpp 3226 2011-07-01 09:48:33Z aguinet $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 */

#include <pvkernel/rush/PVNraw.h>
#include <unistd.h>
#include <tbb/tick_count.h>
#include <iostream>

#define DEFAULT_LINE_SIZE 100
#define MAX_SIZE_RESERVE (size_t(1024*1024*1024u)) // 1GB

PVRush::PVNraw::PVNraw()
{
	_real_nrows = 0;
	_ncols_reserved = 0;
	_chunks_extract = new list_chunks_t();
	_reallocated_buffers = new PVCore::buf_list_t();
}

PVRush::PVNraw::~PVNraw()
{
	clear();
	delete _chunks_extract;
	delete _reallocated_buffers;
}

void PVRush::PVNraw::reserve(PVRow row, PVCol col)
{
	clear_table();
	if (col == 0) {
		col = 1;
	}
	_ncols_reserved = col;
	/*
#ifndef WIN32
	size_t max_alloc = (sysconf(_SC_AVPHYS_PAGES) * sysconf(_SC_PAGE_SIZE))/10;
	if (max_alloc > 0 && row*col*sizeof(PVCore::PVUnicodeString) > max_alloc) {
		row = max_alloc/(col*sizeof(PVCore::PVUnicodeString));
	}
#endif
	table.resize(row, col);
	*/
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
	_ncols_reserved = 0;
	table.clear();
	list_chunks_t::iterator it;
	for (it = _chunks_extract->begin(); it != _chunks_extract->end(); it++) {
		(*it)->free();
	}
	_chunks_extract->clear();

	{
		static tbb::scalable_allocator<char> alloc;
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

	list_chunks_t* ltmp = dst._chunks_extract;
	dst._chunks_extract = src._chunks_extract;
	src._chunks_extract = ltmp;

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
	if (_chunks_extract->size() == 0) {
		return;
	}
	if (_real_nrows > PICVIZ_LINES_MAX) {
		_real_nrows = PICVIZ_LINES_MAX;
	}
	if (_ncols_reserved == 0) {
		// Get the number of fields of the first element
		_ncols_reserved = _chunks_extract->front()->c_elements().front()->c_fields().size();
	}
	PVLOG_INFO("(PVNraw::fit_to_content) nrows=%u, ncols=%u\n", _real_nrows, _ncols_reserved);
	table.resize(_real_nrows, _ncols_reserved);
	list_chunks_t::iterator it_chunk;
	PVRow cur_row = 0;
	for (it_chunk = _chunks_extract->begin(); it_chunk != _chunks_extract->end(); it_chunk++) {
		PVCore::PVChunk* c = *it_chunk;
		PVCore::list_elts::iterator it_elt;
		for (it_elt = c->elements().begin(); it_elt != c->elements().end(); it_elt++) {
			PVCore::PVElement& e = *(*it_elt);
			if (!e.valid())
				continue;
			PVCore::list_fields const& fields = e.c_fields();
			if (fields.size() == 0)
				continue;

			set_row(cur_row, e);
			cur_row++;
		}
	}
	assert(cur_row == _real_nrows);

	PVLOG_INFO("(PVNraw::fit_to_content) done.\n");
}

void PVRush::PVNraw::dump_csv()
{
	for (PVRow i = 0; i < table.get_nrows(); i++) {
		PVLOG_DEBUG("%s\n", qPrintable(nraw_line_to_csv(i)));
	}
#if 0
	PVRush::PVNraw::nraw_table &nraw = get_table();
	PVRush::PVNraw::nraw_table::iterator it_nraw;
	PVRush::PVNraw::nraw_table_line::iterator it_nraw_line, it_nraw_line_end;
	for (it_nraw = nraw.begin(); it_nraw != nraw.end(); it_nraw++) {
		PVRush::PVNraw::nraw_table_line &l = *it_nraw
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
