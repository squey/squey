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

#include <chrono>
#include <memory>

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/filter/PVElementFilterByFields.h>
#include <pvkernel/rush/PVUtils.h>
#include <pvkernel/core/squey_assert.h>

#include "helpers.h"
#include "common.h"

#ifdef SQUEY_BENCH
constexpr static size_t nb_dup = 20;
#else
constexpr static size_t nb_dup = 1;
#endif

int main(int argc, char** argv)
{
	const char* log_file = TEST_FOLDER "/pvkernel/rush/splitters/csv/proxy_sample.csv";
#ifndef SQUEY_BENCH
	const char* ref_file = TEST_FOLDER "/pvkernel/rush/splitters/csv/proxy_sample.csv.out";
#endif

	size_t n = 15;
	if (argc == 4) {
		log_file = argv[1];
#ifndef SQUEY_BENCH
		ref_file = argv[2];
#endif
		n = std::atoi(argv[3]);
	}

	pvtest::TestSplitter ts(log_file, nb_dup);

	// Prepare splitter plugin
	PVFilter::PVFieldsSplitter::p_type sp_lib_p =
	    LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_class_by_name("csv");
	sp_lib_p->set_number_expected_fields(n);

	PVCore::PVArgumentList args = sp_lib_p->get_args();
	args["sep"] = QVariant(QChar(','));
	sp_lib_p->set_args(args);

	auto ff =
	    std::make_unique<PVFilter::PVElementFilterByFields>();
	ff->add_filter(sp_lib_p);
	PVFilter::PVChunkFilterByElt chk_flt{std::move(ff)};

	auto res = ts.run_normalization(chk_flt);
	std::string output_file = std::get<2>(res);
	size_t nelts_org = std::get<0>(res);
	size_t nelts_valid = std::get<1>(res);

	PV_VALID(nelts_valid, nelts_org);

#ifndef SQUEY_BENCH
	std::cout << std::endl << output_file << " - " << ref_file << std::endl;
	// Check output is the same as the reference
	PV_ASSERT_VALID(PVRush::PVUtils::files_have_same_content(output_file, ref_file));
#endif
	std::remove(output_file.c_str());

	return 0;
}
