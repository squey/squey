//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include "PVDBSource.h"
#include <pvkernel/core/PVTextChunk.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>

#include <QSqlError>
#include <QSqlRecord>

#define NCHUNKS 100

PVRush::PVDBSource::PVDBSource(PVDBQuery const& query, chunk_index nelts_chunk)
    : _query(query), _nelts_chunk(nelts_chunk)
{
	seek_begin();
	if (!_query.connect_serv()) {
		PVLOG_WARN("Unable to connect to database: %s\n", qPrintable(_query.last_error_serv()));
	}
	prepare_for_nelts(nelts_chunk * NCHUNKS);
	_last_elt_index = 0;
}

PVRush::PVDBSource::~PVDBSource() = default;

QString PVRush::PVDBSource::human_name()
{
	return _query.human_name();
}

void PVRush::PVDBSource::seek_begin()
{
	_start = 0;
	_next_index = 0;
}

void PVRush::PVDBSource::prepare_for_nelts(chunk_index nelts)
{
	PVLOG_DEBUG("(PVDBSource::prepare_for_nelts) prepare for %d elts\n", nelts);
	_min_nelts = nelts;
	_sql_query = _query.to_query(_start, _min_nelts);
}

PVCore::PVTextChunk* PVRush::PVDBSource::operator()()
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
	PVCore::PVTextChunk* chunk = PVCore::PVTextChunkMem<>::allocate(0, this);
	size_t chunk_size = 0;
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
			chunk_size += value.size();
			memcpy(f.begin(), value.c_str(), value.size());
			elt->fields().push_back(f);
		}
	}
	chunk->set_init_size(chunk_size);

	// Compute the next chunk's index
	_next_index += chunk->c_elements().size();
	if (_next_index - 1 > _last_elt_index) {
		_last_elt_index = _next_index - 1;
	}

	return chunk;
}
