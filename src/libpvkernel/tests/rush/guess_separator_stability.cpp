/* * MIT License
 *
 * © Squey, 2026
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

// A comma-separated file whose last field is a date is split just as evenly by
// the space as by the comma: both are regular on every line, so both match the
// same number of times. The winner used to be whichever Qt's hash table yielded
// first, and Qt seeds that order afresh on every run -- the same file came out
// four columns wide one run and two the next.
//
// CTest runs this under several QT_HASH_SEED values: the answer has to be the
// same under all of them, and it has to be the comma, which tells more columns
// apart than the space does.

#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/filter/PVFieldSplitterChunkMatch.h>
#include <pvkernel/core/squey_assert.h>

#include "common.h"

#include <fstream>
#include <string>

int main()
{
	pvtest::init_ctxt();

	const std::string path =
	    PVRush::PVNrawCacheManager::nraw_dir().toStdString() + "/guess_separator.csv";
	{
		std::ofstream out(path);
		for (int i = 1; i <= 8; i++) {
			out << "host" << i << ",4" << i << ",1." << i << ",2026-03-0" << i
			    << " 10:0" << i << ":00\n";
		}
	}

	PVRush::PVInputDescription_p file(new PVRush::PVFileDescription(QString::fromStdString(path)));
	PVRush::PVSourceCreator_p sc =
	    LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("text_file");
	PVRush::PVSourceCreator::source_p src = sc->create_source_from_input(file);
	PV_ASSERT_VALID(src.get() != nullptr);

	PVCol nfields(0);
	PVFilter::PVFieldsSplitter_p splitter =
	    PVFilter::PVFieldSplitterChunkMatch::get_match_on_input(src, nfields);

	PV_ASSERT_VALID(splitter.get() != nullptr);
	PV_VALID(splitter->registered_name().toStdString(), std::string("csv"));
	PV_VALID(splitter->get_args().at("sep").toString().toStdString(), std::string(","));
	PV_VALID((size_t)nfields, size_t(4));

	return 0;
}
