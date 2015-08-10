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
#include <thread>

#include <tbb/pipeline.h>

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

	int request_count = std::thread::hardware_concurrency();

	PVCore::PVChunk* chunk;

	using indexed_json_buffer_t = std::pair<std::string, size_t>;

	std::vector<PVElasticsearchAPI::rows_t> rows_array;
	rows_array.resize(request_count);

	tbb::parallel_pipeline(request_count,
		tbb::make_filter<void, indexed_json_buffer_t>(tbb::filter::serial_in_order,
			[&](tbb::flow_control& fc)
			{
				if (--request_count == -1 || _query_end) {
					fc.stop();
					return indexed_json_buffer_t();
				}
				std::string json_buffer;

				_elasticsearch.scroll(_query, json_buffer);

				return indexed_json_buffer_t(std::move(json_buffer), request_count);
			}
		) &
		tbb::make_filter<indexed_json_buffer_t, void>(tbb::filter::parallel,
			[&](indexed_json_buffer_t json_buffer)
			{
				if(_elasticsearch.parse_results(json_buffer.first, rows_array[json_buffer.second]) == false) {
					_query_end = true;
				}
			}
		)
	);

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
