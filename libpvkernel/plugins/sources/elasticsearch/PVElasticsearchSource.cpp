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

PVCore::PVChunk* PVRush::PVElasticsearchSource::operator()()
{
	if (_query_end) {
		_query_end = false;

		_elasticsearch.clear_scroll();

		return nullptr;
	}

	PVElasticsearchAPI::rows_chunk_t rows_array;
	_query_end = _elasticsearch.extract(_query, rows_array);

	PVCore::PVChunk* chunk;

	chunk = PVCore::PVChunkMem<>::allocate(0, this);
	chunk->set_index(_next_index);

	for (const PVElasticsearchAPI::rows_t& rows : rows_array) {
		for (const std::string& row : rows) {
			PVCore::PVElement* elt = chunk->add_element();
			elt->fields().clear();

			PVCore::PVField f(*elt);
			f.allocate_new(row.size());
			memcpy(f.begin(), row.c_str(), row.size());
			elt->fields().push_back(f);
		}
	}

	// Compute the next chunk's index
	_next_index += chunk->c_elements().size();
	if (_next_index - 1 > _last_elt_index) {
		_last_elt_index = _next_index - 1;
	}

	return chunk;
}
