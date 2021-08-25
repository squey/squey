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

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/filter/PVElementFilterByFields.h>
#include <pvkernel/filter/PVFieldsMappingFilter.h>
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVUtils.h>
#include <pvkernel/core/inendi_assert.h>

#include <iostream>

#include "common.h"
#include "helpers.h"

static constexpr const char* log_file = TEST_FOLDER "/pvkernel/rush/filter_composition";
#ifndef INSPECTOR_BENCH
static constexpr const char* ref_file = TEST_FOLDER "/pvkernel/rush/filter_composition.out";
#endif

#ifdef INSPECTOR_BENCH
constexpr static size_t nb_dup = 1000;
#else
constexpr static size_t nb_dup = 1;
#endif

using namespace PVRush;
using namespace PVCore;

int main()
{
	pvtest::TestSplitter ts(log_file, nb_dup);

	PVFilter::PVFieldsSplitter::p_type url_lib_p =
	    LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_class_by_name("url");
	PVFilter::PVFieldsSplitter::p_type regexp_lib_p =
	    LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_class_by_name("regexp");
	PVFilter::PVFieldsSplitter::p_type duplicate_lib_p =
	    LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_class_by_name("duplicate");
	PVFilter::PVFieldsFilter<PVFilter::one_to_one>::p_type grep_lib_p =
	    LIB_CLASS(PVFilter::PVFieldsFilter<PVFilter::one_to_one>)::get().get_class_by_name(
	        "regexp");

	PVCore::PVArgumentList args;
	args["regexp"] = QString("([0-9]+)[0-9.]*\\s+[0-9]+\\s+[0-9]+\\s+[A-Z/"
	                         "_-]+([0-9]+)\\s+[0-9]+\\s+(GET|POST|PUT|OPTIONS)\\s+(\\S+)\\s+(\\S+)"
	                         "\\s+([^/]+)/(\\d+.\\d+.\\d+.\\d+)");
	args["full-line"] = false;
	regexp_lib_p->set_args(args);
	args["regexp"] = QString("(yahoo|lnc)");
	args["reverse"] = true;
	grep_lib_p->set_args(args);
	args.clear();
	args["n"] = 4;
	duplicate_lib_p->set_args(args);

	// Mapping filters

	// Mapping filter for the URL splitter
	PVFilter::PVFieldsBaseFilter_p mapping_url(new PVFilter::PVFieldsMappingFilter(3, url_lib_p));

	// Mapping filter for the grep filter
	PVFilter::PVFieldsBaseFilter_p mapping_grep(new PVFilter::PVFieldsMappingFilter(3, grep_lib_p));

	// Mapping filter for the duplicate filter on the last axis after our regexp
	PVFilter::PVFieldsBaseFilter_p mapping_duplicate(
	    new PVFilter::PVFieldsMappingFilter(6, duplicate_lib_p));

	// Final composition
	auto ff =
	    std::unique_ptr<PVFilter::PVElementFilterByFields>(new PVFilter::PVElementFilterByFields());
	ff->add_filter(std::move(regexp_lib_p));
	ff->add_filter(std::move(mapping_duplicate));
	ff->add_filter(std::move(mapping_grep));
	ff->add_filter(std::move(mapping_url));

	PVFilter::PVChunkFilterByElt chk_flt{std::move(ff)};

	auto res = ts.run_normalization(chk_flt);
	std::string output_file = std::get<2>(res);
	size_t nelts_org = std::get<0>(res);
	size_t nelts_valid = std::get<1>(res);

	PV_VALID(nelts_valid, 760UL * nb_dup);
	PV_VALID(nelts_org, 1000UL * nb_dup);

#ifndef INSPECTOR_BENCH
	// Check output is the same as the reference
	std::cout << std::endl << output_file << " - " << ref_file << std::endl;
	PV_ASSERT_VALID(PVRush::PVUtils::files_have_same_content(output_file, ref_file));
#endif
	std::remove(output_file.c_str());
}
