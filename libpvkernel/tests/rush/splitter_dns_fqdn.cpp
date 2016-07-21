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
constexpr static size_t nb_dup = 1000;
#else
constexpr static size_t nb_dup = 1;
#endif

static constexpr const char* log_file = TEST_FOLDER "/pvkernel/rush/splitters/fqdn/fqnd.log";
#ifndef INSPECTOR_BENCH
static constexpr const char* ref_file = TEST_FOLDER "/pvkernel/rush/splitters/fqdn/fqnd.log.out";
#endif

int main()
{
	pvtest::TestSplitter ts(log_file, nb_dup);

	// Prepare splitter plugin
	PVFilter::PVFieldsSplitter::p_type sp_lib_p =
	    LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_class_by_name("dns_fqdn");

	PVCore::PVArgumentList args = sp_lib_p->get_args();
	args["N"] = 3;
	args["tld1"] = true;
	args["tld2"] = true;
	args["tld3"] = true;
	args["subd1"] = true;
	args["subd2"] = true;
	args["subd3"] = true;
	args["subd1_rev"] = false;
	sp_lib_p->set_args(args);

	PVFilter::PVChunkFilterByElt chk_flt{
	    std::unique_ptr<PVFilter::PVElementFilterByFields>(new PVFilter::PVElementFilterByFields(
	        [&](PVCore::list_fields& fields) -> PVCore::list_fields& {
		        return (*sp_lib_p)(fields);
		    }))};

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
