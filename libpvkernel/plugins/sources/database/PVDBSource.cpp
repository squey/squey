/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVDBSource.h"
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>

#include <QSqlError>
#include <QSqlRecord>

#define NCHUNKS 100

PVRush::PVDBSource::PVDBSource(PVDBQuery const& query, chunk_index nelts_chunk)
    : PVRawSourceBase(), _query(query), _nelts_chunk(nelts_chunk)
{
	seek_begin();
	if (!_query.connect_serv()) {
		PVLOG_WARN("Unable to connect to database: %s\n", qPrintable(_query.last_error_serv()));
	}
	prepare_for_nelts(nelts_chunk * NCHUNKS);
	_last_elt_index = 0;
}

PVRush::PVDBSource::~PVDBSource()
{
}

QString PVRush::PVDBSource::human_name()
{
	return _query.human_name();
}

void PVRush::PVDBSource::seek_begin()
{
	seek(0);
}

bool PVRush::PVDBSource::seek(input_offset off)
{
	_start = off;
	_next_index = off;
	return true;
}

void PVRush::PVDBSource::prepare_for_nelts(chunk_index nelts)
{
	PVLOG_DEBUG("(PVDBSource::prepare_for_nelts) prepare for %d elts\n", nelts);
	_min_nelts = nelts;
	_sql_query = _query.to_query(_start, _min_nelts);
}

PVCore::PVChunk* PVRush::PVDBSource::operator()()
{
	if (!_sql_query.isActive()) {
		if (!_sql_query.exec()) {
			PVLOG_WARN("(PVDBSource::operator()) unable to exec SQL query '%s': %s.\n",
			           qPrintable(_sql_query.lastQuery()),
			           qPrintable(_sql_query.lastError().text()));
			return nullptr;
		}

		_sql_query.seek(_start);
		PVLOG_DEBUG("(PVDBSource::operator()) executed query '%s'.\n",
		            qPrintable(_sql_query.lastQuery()));
	}
	// Create a chunk w/ no memory for its internal buffer
	PVCore::PVChunk* chunk = PVCore::PVChunkMem<>::allocate(0, this);
	chunk->set_index(_next_index);
	PVLOG_INFO("_nelts_chunk=%d; _next_index=%d\n", _nelts_chunk, _next_index);
	for (chunk_index n = 0; n < _nelts_chunk; n++) {
		if (!_sql_query.next()) {
			if (n == 0) {
				return nullptr;
			}
			break;
		}
		QSqlRecord rec = _sql_query.record();
		PVCore::PVElement* elt = chunk->add_element();
		elt->fields().clear();
		for (int i = 0; i < rec.count(); i++) {
			std::string value = rec.value(i).toString().toStdString();
			PVCore::PVField f(*elt);
			f.allocate_new(value.size());
			memcpy(f.begin(), value.c_str(), value.size());
			elt->fields().push_back(f);
		}
	}

	// Set the index of the elements inside the chunk
	chunk->set_elements_index();

	// Compute the next chunk's index
	_next_index += chunk->c_elements().size();
	if (_next_index - 1 > _last_elt_index) {
		_last_elt_index = _next_index - 1;
	}

	return chunk;
}
