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

#include "PVSplunkSource.h"
#include "../../common/splunk/PVSplunkQuery.h"

#include <sstream>

PVRush::PVSplunkSource::PVSplunkSource(PVInputDescription_p input)
    : _next_index(0)
    , _query(*dynamic_cast<PVSplunkQuery*>(input.get()))
    , _splunk(_query.get_infos())
{
}

PVRush::PVSplunkSource::~PVSplunkSource() = default;

QString PVRush::PVSplunkSource::human_name()
{
	return {"splunk"};
}

void PVRush::PVSplunkSource::seek_begin() {}

void PVRush::PVSplunkSource::prepare_for_nelts(chunk_index /*nelts*/) {}

PVCore::PVTextChunk* PVRush::PVSplunkSource::operator()()
{
	if (_query_end) {
		_query_end = false;

		return nullptr;
	}

	std::string data;

	_query_end = !_splunk.extract(_query, data);

	PVCore::PVTextChunk* chunk;

	size_t chunk_size = 0;
	chunk = PVCore::PVTextChunkMem<>::allocate(0, this);
	chunk->set_index(_next_index);

	std::istringstream iss(data);
	for (std::string row; std::getline(iss, row);) {
		PVCore::PVElement* elt = chunk->add_element();
		elt->fields().clear();

		PVCore::PVField f(*elt);
		f.allocate_new(row.size());
		chunk_size += row.size();
		memcpy(f.begin(), row.c_str(), row.size());
		elt->fields().push_back(f);
	}
	chunk->set_init_size(chunk_size);

	// Compute the next chunk's index
	_next_index += chunk->c_elements().size();
	if (_next_index - 1 > _last_elt_index) {
		_last_elt_index = _next_index - 1;
	}

	return chunk;
}
