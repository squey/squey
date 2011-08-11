#include "PVDBSource.h"
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>

#include <QSqlError>
#include <QSqlRecord>

#define NCHUNKS 100

PVRush::PVDBSource::PVDBSource(PVDBQuery const& query, chunk_index nelts_chunk, PVFilter::PVChunkFilter_f src_filter):
	PVRawSourceBase(src_filter),
	_query(query),
	_nelts_chunk(nelts_chunk)
{
	seek(0);
	if (!_query.connect_serv()) {
		PVLOG_WARN("Unable to connect to database: %s\n", qPrintable(_query.last_error_serv()));
	}
	prepare_for_nelts(nelts_chunk*NCHUNKS);
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
			PVLOG_WARN("(PVDBSource::operator()) unable to exec SQL query '%s': %s.\n", qPrintable(_sql_query.lastQuery()), qPrintable(_sql_query.lastError().text()));
			return NULL;
		}
		PVLOG_DEBUG("(PVDBSource::operator()) executed query '%s'.\n", qPrintable(_sql_query.lastQuery()));
	}
	// Create a chunk w/ no memory for its internal buffer
	PVCore::PVChunk* chunk = PVCore::PVChunkMem<>::allocate(0, this);
	chunk->set_index(_next_index);
	for (chunk_index n = 0; n < _nelts_chunk; n++) {
		if (!_sql_query.next()) {
			// Try to get the next batch, and if empty, well, that's the end.
			query_next_batch();
			if (_sql_query.size() == 0) {
				if (n == 0) {
					return NULL;
				}
				break;
			}
		}
		PVCore::PVElement elt(chunk);
		elt.fields().clear();
		QSqlRecord rec = _sql_query.record();
		for (int i = 0; i < rec.count(); i++) {
			QString value = rec.value(i).toString();
			PVCore::PVField f(elt);
			size_t size_buf = value.size() * sizeof(QChar);
			f.allocate_new(size_buf);
			memcpy(f.begin(), value.constData(), size_buf);
			elt.fields().push_back(f);
		}
		chunk->elements().push_back(elt);
	}

	// Set the index of the elements inside the chunk
	chunk->set_elements_index();

	// Compute the next chunk's index
	_next_index += chunk->c_elements().size();
	if (_next_index-1>_last_elt_index) {
		_last_elt_index = _next_index-1;
	}

	return chunk;
}

void PVRush::PVDBSource::query_next_batch()
{
	// Ask for the next batch.
	// We ask lines for creating 'NCHUNKS' chunk of '_nelts_chunk' elements
	seek(_start + _min_nelts + 1);
	prepare_for_nelts(_nelts_chunk);
	_sql_query.exec();
	PVLOG_DEBUG("(PVDBSource::query_next_batch) executed query '%s'.\n", qPrintable(_sql_query.lastQuery()));
}

PVRush::input_offset PVRush::PVDBSource::get_input_offset_from_index(chunk_index idx, chunk_index& known_idx)
{
	// This is an identity function, as elements' index, in the
	// case of DB, are row indexes.
	known_idx = idx;
	return idx;
}
