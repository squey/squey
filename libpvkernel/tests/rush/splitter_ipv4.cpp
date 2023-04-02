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

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/filter/PVElementFilterByFields.h>
#include <pvkernel/rush/PVUtils.h>
#include <pvkernel/core/inendi_assert.h>

#include "helpers.h"
#include "common.h"

#ifdef INSPECTOR_BENCH
constexpr static size_t nb_dup = 1000;
#else
constexpr static size_t nb_dup = 1;
#endif

static constexpr const char* log_file = TEST_FOLDER "/pvkernel/rush/splitters/ip/ipv4";
#ifndef INSPECTOR_BENCH
static constexpr const char* ref_file = TEST_FOLDER "/pvkernel/rush/splitters/ip/ipv4.out";
#endif

// this variable must not be defined as const
std::string value = "aa.bb.cc.dd";

using string_vector = std::vector<std::string>;

const string_vector valid_params_list = {"0", "1", "2", "0,1", "0,2", "1,2", "0,1,2"};
const std::vector<string_vector> valid_params_result = {
    {"aa", "bb.cc.dd"},    {"aa.bb", "cc.dd"},    {"aa.bb.cc", "dd"},      {"aa", "bb", "cc.dd"},
    {"aa", "bb.cc", "dd"}, {"aa.bb", "cc", "dd"}, {"aa", "bb", "cc", "dd"}};

const string_vector invalid_params_list = {"3", "1,3", "1,3,2"};

void check_valid_params(PVFilter::PVFieldsSplitter::p_type& splitter,
                        const std::string& params,
                        const string_vector& expected)
{
	PVCore::PVArgumentList args = splitter->get_args();

	args["ipv6"] = false;
	args["params"] = params.c_str();

	try {
		splitter->set_args(args);
	} catch (PVFilter::PVFieldsFilterInvalidArguments& e) {
		std::cerr << "valid value \"" << params << "\" for 'params' is considered invalid"
		          << std::endl;
		throw e;
	}

	PVCore::list_fields res_list;

	PVCore::PVElement in_elt(nullptr);
	char* value_str = &value[0];

	PVCore::PVField in_field(in_elt, value_str, value_str + value.size());
	PVCore::list_fields in_fields_list;

	in_fields_list.push_back(in_field);

	res_list = splitter->operator()(in_fields_list);

	PV_VALID(res_list.size(), expected.size());

	size_t i = 0;
	for (auto& f : res_list) {
		std::string v(f.begin(), f.end());
		PV_VALID(v, expected[i]);
		++i;
	}
}

void check_invalid_params(PVFilter::PVFieldsSplitter::p_type& splitter, const std::string& params)
{
	PVCore::PVArgumentList args = splitter->get_args();

	args["ipv6"] = false;
	args["params"] = params.c_str();
	;
	bool has_invalid_params = false;

	try {
		splitter->set_args(args);
	} catch (PVFilter::PVFieldsFilterInvalidArguments&) {
		has_invalid_params = true;
	}

	PV_VALID(has_invalid_params, true);
}

int main()
{
	pvtest::TestSplitter ts(log_file, nb_dup);

	// Prepare splitter plugin
	PVFilter::PVFieldsSplitter::p_type sp_lib_p =
	    LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_class_by_name("ip");

#ifndef INSPECTOR_BENCH
	for (size_t i = 0; i < valid_params_list.size(); ++i) {
		check_valid_params(sp_lib_p, valid_params_list[i], valid_params_result[i]);
	}

	for (const auto & i : invalid_params_list) {
		check_invalid_params(sp_lib_p, i);
	}
#endif

	PVCore::PVArgumentList args = sp_lib_p->get_args();
	args["ipv6"] = false;
	args["params"] = "1,2";
	sp_lib_p->set_args(args);

	auto ff =
	    std::unique_ptr<PVFilter::PVElementFilterByFields>(new PVFilter::PVElementFilterByFields());
	ff->add_filter(sp_lib_p);
	PVFilter::PVChunkFilterByElt chk_flt{std::move(ff)};

	auto res = ts.run_normalization(chk_flt);
	std::string output_file = std::get<2>(res);
	size_t nelts_org = std::get<0>(res);
	size_t nelts_valid = std::get<1>(res);

	PV_VALID(nelts_valid, nelts_org);

#ifndef INSPECTOR_BENCH
	// Check output is the same as the reference
	std::cout << std::endl << output_file << " - " << ref_file << std::endl;
	PV_ASSERT_VALID(PVRush::PVUtils::files_have_same_content(output_file, ref_file));
#endif
	std::remove(output_file.c_str());
	return 0;
}
