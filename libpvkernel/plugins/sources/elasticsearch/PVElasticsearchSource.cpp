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

#include <pvkernel/core/PVTextChunk.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>

#include <pvkernel/rush/PVFileDescription.h>

#include "PVElasticsearchSource.h"
#include "../../common/elasticsearch/PVElasticsearchQuery.h"

#include <string>

static constexpr const size_t MEGA = 1024 * 1024;

PVRush::PVElasticsearchSource::PVElasticsearchSource(PVInputDescription_p input)
    : _next_index(0)
    , _query(*dynamic_cast<PVElasticsearchQuery*>(input.get()))
    , _elasticsearch(_query.get_infos())
{
	const PVElasticsearchInfos& infos = _query.get_infos();

	PVLOG_INFO("Create elasticsearch source with: host=%s, port=%d, index=%s\n",
	           qPrintable(infos.get_host()), infos.get_port(), qPrintable(infos.get_index()));
}

PVRush::PVElasticsearchSource::~PVElasticsearchSource() {}

QString PVRush::PVElasticsearchSource::human_name()
{
	return QString("elasticsearch");
}

void PVRush::PVElasticsearchSource::seek_begin() {}

void PVRush::PVElasticsearchSource::prepare_for_nelts(chunk_index /*nelts*/) {}

size_t PVRush::PVElasticsearchSource::get_size() const
{
	/*
	 *  Progress API expects input size to be given in MB
	 *  but Elasticsearch only return the number of rows,
	 *  so let's pretend it's a MB size...
	 */
	return _elasticsearch.count(_query) * MEGA;
}

PVCore::PVTextChunk* PVRush::PVElasticsearchSource::operator()()
{
	if (_query_end) {
		_query_end = false;

		_elasticsearch.clear_scroll();

		return nullptr;
	}

	PVElasticsearchAPI::rows_t rows;
	_query_end = not _elasticsearch.extract(_query, rows);

	// Create a chunk w/ no memory for its internal buffer
	PVCore::PVTextChunk* chunk = PVCore::PVTextChunkMem<>::allocate(0, this);
	chunk->set_index(_next_index);

	for (const std::vector<std::string>& row : rows) {
		PVCore::PVElement* elt = chunk->add_element();
		elt->fields().clear();
		for (const std::string& field : row) {
			PVCore::PVField f(*elt);
			f.allocate_new(field.size());
			memcpy(f.begin(), field.c_str(), field.size());
			elt->fields().push_back(f);
		}
	}
	chunk->set_init_size(rows.size() * MEGA);

	// Compute the next chunk's index
	_next_index += chunk->c_elements().size();
	if (_next_index - 1 > _last_elt_index) {
		_last_elt_index = _next_index - 1;
	}

	return chunk;
}
