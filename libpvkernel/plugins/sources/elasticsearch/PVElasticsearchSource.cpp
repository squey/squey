/**
 * \file PVElasticsearchSource.cpp
 *
 * Copyright (C) Picviz Labs 2015
 */

#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>

#include <pvkernel/rush/PVFileDescription.h>

#include "PVElasticsearchSource.h"
#include "../../common/elasticsearch/PVElasticsearchQuery.h"

#include <string>

PVRush::PVElasticsearchSource::PVElasticsearchSource(PVInputDescription_p input, PVFilter::PVChunkFilter_f src_filter):
	PVRawSourceBase(src_filter),
	_next_index(0),
	_query(*dynamic_cast<PVElasticsearchQuery*>(input.get())),
	_elasticsearch(_query.get_infos())
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
	seek(0);
}

bool PVRush::PVElasticsearchSource::seek(input_offset /*off*/)
{
	assert(false);
	return false;
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

			QString value(QString::fromUtf8(row.c_str(), row.size()));
			PVCore::PVField f(*elt);
			size_t size_buf = value.size() * sizeof(QChar);
			f.allocate_new(size_buf);
			memcpy(f.begin(), value.constData(), size_buf);
			elt->fields().push_back(f);
		}
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
