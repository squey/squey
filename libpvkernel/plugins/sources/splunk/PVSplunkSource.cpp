/**
 * @file
 *
 * @copyright (C) Picviz Labs 2015
 */

#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>

#include <pvkernel/rush/PVFileDescription.h>

#include "PVSplunkSource.h"
#include "../../common/splunk/PVSplunkQuery.h"

PVRush::PVSplunkSource::PVSplunkSource(PVInputDescription_p input, PVFilter::PVChunkFilter_f src_filter)
	: PVRawSourceBase(src_filter),
	  _query(*dynamic_cast<PVSplunkQuery*>(input.get())),
	  _splunk(_query.get_infos())
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
	seek(0);
}

bool PVRush::PVSplunkSource::seek(input_offset /*off*/)
{
	return false;
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
	size_t data_size;

	_query_end = !_splunk.extract(_query, data, data_size);

	PVCore::PVChunk* chunk;

	chunk = PVCore::PVChunkMem<>::allocate(0, this);
	chunk->set_index(_next_index);

	size_t processed_size = 0;

	std::istringstream iss(data);
	for (std::string row; std::getline(iss, row) && processed_size < data_size;) {
		PVCore::PVElement* elt = chunk->add_element();
		elt->fields().clear();

		QString value(QString::fromUtf8(row.c_str(), row.size()));
		PVCore::PVField f(*elt);
		size_t size_buf = value.size() * sizeof(QChar);
		f.allocate_new(size_buf);
		memcpy(f.begin(), value.constData(), size_buf);
		elt->fields().push_back(f);

		processed_size += row.size()+1; // +1 to count std::end that std::getline discards
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
