/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <chrono>

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/filter/PVElementFilterByFields.h>
#include <pvkernel/rush/PVUtils.h>
#include <pvkernel/core/inendi_assert.h>

#include "helpers.h"
#include "common.h"

#ifdef INSPECTOR_BENCH
constexpr static size_t nb_dup = 10000;
#else
constexpr static size_t nb_dup = 1;
#endif

static constexpr const char* log_file = TEST_FOLDER "/pvkernel/rush/splitters/duplicate/lines.txt";
#ifndef INSPECTOR_BENCH
static constexpr const char* ref_file =
    TEST_FOLDER "/pvkernel/rush/splitters/duplicate/lines.txt.out";
#endif

int main()
{
	pvtest::TestSplitter ts(log_file, nb_dup);

	// Prepare splitter plugin
	PVFilter::PVFieldsSplitter::p_type sp_lib_p =
	    LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_class_by_name("duplicate");

	PVCore::PVArgumentList args = sp_lib_p->get_args();
	args["n"] = 7;
	sp_lib_p->set_args(args);

	PVFilter::PVElementFilterByFields* elt_f = new PVFilter::PVElementFilterByFields(sp_lib_p->f());
	PVFilter::PVChunkFilterByElt* chk_flt = new PVFilter::PVChunkFilterByElt(elt_f->f());
	PVFilter::PVChunkFilter_f flt_f = chk_flt->f();

	auto res = ts.run_normalization(flt_f);

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
