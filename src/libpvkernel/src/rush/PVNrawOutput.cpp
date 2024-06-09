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

#include <pvkernel/rush/PVNraw.h> // for PVNraw
#include <pvkernel/rush/PVNrawOutput.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/core/PVTextChunk.h>
#include <pvkernel/core/PVBinaryChunk.h>
#include <pvbase/types.h> // for PVRow
#include <assert.h>
#include <cstddef> // for size_t
#include <map>     // for map
#include <string>  // for string
#include <atomic>

#include "pvkernel/core/PVChunk.h"

PVRush::PVNrawOutput::PVNrawOutput(PVNraw& nraw) : _nraw_dest(&nraw) {}

void PVRush::PVNrawOutput::operator()(PVCore::PVChunk* out)
{
	if (auto* text_chunk = dynamic_cast<PVCore::PVTextChunk*>(out)) {
		nraw_dest().add_chunk_utf16(*text_chunk);
	} else {
		assert(dynamic_cast<PVCore::PVBinaryChunk*>(out));
		auto* bin_chunk = static_cast<PVCore::PVBinaryChunk*>(out);

		nraw_dest().add_bin_chunk(*bin_chunk);
	}

	_out_size += out->get_init_size();

	// Clear this chunk !
	out->free();
}

PVRow PVRush::PVNrawOutput::get_rows_count()
{
	if (_nraw_dest != nullptr) {
		return _nraw_dest->row_count();
	} else {
		return 0;
	}
}

void PVRush::PVNrawOutput::prepare_load(const PVRush::PVFormat& format)
{
	_nraw_dest->prepare_load(format.get_storage_format());
}

void PVRush::PVNrawOutput::job_has_finished(const std::map<size_t, std::string>& inv_elts)
{
	// Tell the destination NRAW that clean up can be done, everything is imported
	nraw_dest().load_done(inv_elts);
}
