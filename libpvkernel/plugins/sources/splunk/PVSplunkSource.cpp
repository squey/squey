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

#include "PVSplunkSource.h"
#include "../../common/splunk/PVSplunkQuery.h"

PVRush::PVSplunkSource::PVSplunkSource(PVInputDescription_p input)
    : PVRawSourceBase()
    , _next_index(0)
    , _query(*dynamic_cast<PVSplunkQuery*>(input.get()))
    , _splunk(_query.get_infos())
{
}

PVRush::PVSplunkSource::~PVSplunkSource()
{
}

QString PVRush::PVSplunkSource::human_name()
{
	return QString("splunk");
}

void PVRush::PVSplunkSource::seek_begin()
{
}

void PVRush::PVSplunkSource::prepare_for_nelts(chunk_index /*nelts*/)
{
}

PVCore::PVChunk* PVRush::PVSplunkSource::operator()()
{
	if (_query_end) {
		_query_end = false;

		return nullptr;
	}

	std::string data;

	_query_end = !_splunk.extract(_query, data);

	PVCore::PVChunk* chunk;

	chunk = PVCore::PVChunkMem<>::allocate(0, this);
	chunk->set_index(_next_index);

	std::istringstream iss(data);
	for (std::string row; std::getline(iss, row);) {
		PVCore::PVElement* elt = chunk->add_element();
		elt->fields().clear();

		PVCore::PVField f(*elt);
		f.allocate_new(row.size());
		memcpy(f.begin(), row.c_str(), row.size());
		elt->fields().push_back(f);
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
