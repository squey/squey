/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2015-2015
 */

#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>

#include <pvkernel/rush/PVFileDescription.h>

#include "PVElasticsearchSource.h"
#include "../../common/elasticsearch/PVElasticsearchQuery.h"

#include <string>

static constexpr const size_t MEGA = 1024 * 1024;

PVRush::PVElasticsearchSource::PVElasticsearchSource(PVInputDescription_p input)
    : PVRawSourceBase()
    , _next_index(0)
    , _query(*dynamic_cast<PVElasticsearchQuery*>(input.get()))
    , _elasticsearch(_query.get_infos())
{
	const PVElasticsearchInfos& infos = _query.get_infos();

	PVLOG_INFO("Create elasticsearch source with: host=%s, port=%d, index=%s\n",
	           qPrintable(infos.get_host()), infos.get_port(), qPrintable(infos.get_index()));
}

PVRush::PVElasticsearchSource::~PVElasticsearchSource()
{
}

QString PVRush::PVElasticsearchSource::human_name()
{
	return QString("elasticsearch");
}

void PVRush::PVElasticsearchSource::seek_begin()
{
}

void PVRush::PVElasticsearchSource::prepare_for_nelts(chunk_index /*nelts*/)
{
}

size_t PVRush::PVElasticsearchSource::get_size() const
{
	/*
	 *  Progress API expects input size to be given in MB
	 *  but Elasticsearch only return the number of rows,
	 *  so let's pretend it's a MB size...
	 */
	return _elasticsearch.count(_query) * MEGA;
}

PVCore::PVChunk* PVRush::PVElasticsearchSource::operator()()
{
	if (_query_end) {
		_query_end = false;

		_elasticsearch.clear_scroll();

		return nullptr;
	}

	PVElasticsearchAPI::rows_t rows;
	_query_end = not _elasticsearch.extract(_query, rows);

	// Create a chunk w/ no memory for its internal buffer
	PVCore::PVChunk* chunk = PVCore::PVChunkMem<>::allocate(0, this);
	size_t chunk_size = 0;
	chunk->set_index(_next_index);

	for (const std::vector<std::string>& row : rows) {
		PVCore::PVElement* elt = chunk->add_element();
		elt->fields().clear();
		for (const std::string& field : row) {
			PVCore::PVField f(*elt);
			f.allocate_new(field.size());
			chunk_size += field.size();
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
