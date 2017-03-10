/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include <chrono>

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/filter/PVElementFilterByFields.h>
#include <pvkernel/rush/PVUtils.h>
#include <pvkernel/core/PVUtils.h>
#include <pvkernel/core/inendi_assert.h>

#include "helpers.h"
#include "common.h"

#ifdef INSPECTOR_BENCH
constexpr static size_t nb_dup = 100000;
#else
constexpr static size_t nb_dup = 1;
#endif

using files_t = std::pair<std::string, std::string>;
using params_t =
    std::tuple<int, bool, std::vector<std::string>, std::string, std::string, bool, char, char>;
using test_t = std::pair<files_t, params_t>;

static const std::vector<test_t> testsuite = {
    // test 1 : whole fields
    {files_t{/* input_file 		*/ TEST_FOLDER "/pvkernel/rush/converter/substitution/input1.log",
             /* ref_file 		*/ TEST_FOLDER "/pvkernel/rush/converter/substitution/ref1.log"},
     params_t{/* modes 			 */ 1,
              /* invert_order 	 */ false,
              /* substrings_map 	 */ {},
              /* path 			 */ TEST_FOLDER "/pvkernel/rush/converter/substitution/map1.csv",
              /* default_value 	 */ "0",
              /* use_default_value */ true,
              /* sep 				 */ ':',
              /* quote 			 */ '"'}},

    // test 2 : substrings
    {files_t{/* input_file 		*/ TEST_FOLDER "/pvkernel/rush/converter/substitution/ref1.log",
             /* ref_file 		*/ TEST_FOLDER "/pvkernel/rush/converter/substitution/ref2.log"},
     params_t{/* modes 			 */ 2,
              /* invert_order 	 */ false,
              /* substrings_map 	 */ {",", ".", " ", ""},
              /* path 			 */ "",
              /* default_value 	 */ "0",
              /* use_default_value */ true,
              /* sep 				 */ ':',
              /* quote 			 */ '"'}},

    // test 3 : whole fields + substrings
    {files_t{/* input_file 		*/ TEST_FOLDER "/pvkernel/rush/converter/substitution/input1.log",
             /* ref_file 		*/ TEST_FOLDER "/pvkernel/rush/converter/substitution/ref2.log"},
     params_t{/* modes 			 */ 3,
              /* invert_order 	 */ false,
              /* substrings_map 	 */ {",", ".", " ", ""},
              /* path 			 */ TEST_FOLDER "/pvkernel/rush/converter/substitution/map1.csv",
              /* default_value 	 */ "0",
              /* use_default_value */ true,
              /* sep 				 */ ':',
              /* quote 			 */ '"'}},

    // test 4 : substrings + whole fields
    {files_t{/* input_file 		*/ TEST_FOLDER "/pvkernel/rush/converter/substitution/ref1.log",
             /* ref_file 		*/ TEST_FOLDER "/pvkernel/rush/converter/substitution/input1.log"},
     params_t{/* modes 			 */ 3,
              /* invert_order 	 */ true,
              /* substrings_map 	 */ {",", ".", " ", ""},
              /* path 			 */ TEST_FOLDER "/pvkernel/rush/converter/substitution/map2.csv",
              /* default_value 	 */ "0",
              /* use_default_value */ true,
              /* sep 				 */ ':',
              /* quote 			 */ '"'}}};

int main()
{
	pvtest::TestSplitter ts;

	// Prepare splitter plugin
	PVFilter::PVFieldsConverter::p_type sp_lib_p =
	    LIB_CLASS(PVFilter::PVFieldsConverter)::get().get_class_by_name("substitution");

	for (const test_t& test : testsuite) {
		const std::string& input_file = test.first.first;
		const std::string& ref_file = test.first.second;

		ts.reset(input_file, nb_dup);

		const params_t& params = test.second;

		PVCore::PVArgumentList args;
		args["modes"] = std::get<0>(params);
		args["invert_order"] = std::get<1>(params);
		QStringList l;
		for (const std::string& s : std::get<2>(params)) {
			l << QString::fromStdString(s);
		};
		args["substrings_map"] = PVCore::serialize_base64(l);
		args["path"] = std::get<3>(params).c_str();
		args["default_value"] = std::get<4>(params).c_str();
		args["use_default_value"] = std::get<5>(params);
		args["sep"] = std::get<6>(params);
		args["quote"] = std::get<7>(params);
		sp_lib_p->set_args(args);

		auto ff = std::unique_ptr<PVFilter::PVElementFilterByFields>(
		    new PVFilter::PVElementFilterByFields());
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
		(void)ref_file;
		std::remove(output_file.c_str());
	}

	return 0;
}
