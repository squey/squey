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

#include <pvkernel/core/inendi_assert.h>

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/filter/PVFieldSplitterChunkMatch.h>

#include "common.h"

#include <iostream>

PVFilter::PVFieldsSplitter_p guess_filter(const char* filename, PVCol& axes_count)
{
	pvtest::init_ctxt();

	// Input file
	QString path_file(filename);
	PVRush::PVInputDescription_p file(new PVRush::PVFileDescription(path_file));

	// Get the source creator
	PVRush::PVSourceCreator_p sc_file =
	    LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("text_file");

	// Process that file with the found source creator thanks to the extractor
	PVRush::PVSourceCreator::source_p src = sc_file->create_source_from_input(file);
	if (!src) {
		return {};
	}

	return PVFilter::PVFieldSplitterChunkMatch::get_match_on_input(src, axes_count);
}

int main(int argc, char** argv)
{
	if (argc != 5) {
		std::cerr << "Usage: " << argv[0] << "fields_count sep quote file" << std::endl;
		return 1;
	}

	PVCol found_fields_count(0);
	PVCol expected_fields_count = (PVCol)atoi(argv[1]);
	QString expected_sep(argv[2]);
	QString expected_quote(argv[3]);

	auto filter = guess_filter(argv[4], found_fields_count);

	// if expected_fields_count == 0, the filter must be null
	if (expected_fields_count == 0) {
		PV_ASSERT_VALID(filter.get() == nullptr);

		return 0;
	}

	/* at that point, the filter must non-null, being just sure it is
	 */
	PV_ASSERT_VALID(filter.get() != nullptr);

	/* checking the found splitter is the CSV one
	 */
	PV_ASSERT_VALID(filter->registered_name() == "csv");

	/* checking the guessed fields count
	 */
	PV_VALID(found_fields_count, expected_fields_count);

	/* checking guessed CSV splitter parameters (seq and quote)
	 */
	const PVCore::PVArgumentList& args = filter->get_args();

	QString found_sep = args.at("sep").toString();
	QString found_quote = args.at("quote").toString();

	PV_ASSERT_VALID(found_sep == expected_sep, "found_sep", found_sep.toStdString(), "expected_sep",
	                expected_sep.toStdString());
	PV_ASSERT_VALID(found_quote == expected_quote, "found_quote", found_quote.toStdString(),
	                "expected_quote", expected_quote.toStdString());

	return 0;
}
