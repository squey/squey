/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

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
static constexpr const char* ref_file = TEST_FOLDER "/pvkernel/rush/filter_composition.out";

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

	PVFilter::PVFieldsSplitter::p_type url_lib_p = LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_class_by_name("url");
	PVFilter::PVFieldsSplitter::p_type regexp_lib_p = LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_class_by_name("regexp");
	PVFilter::PVFieldsSplitter::p_type duplicate_lib_p = LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_class_by_name("duplicate");
	PVFilter::PVFieldsFilter<PVFilter::one_to_one>::p_type grep_lib_p = LIB_CLASS(PVFilter::PVFieldsFilter<PVFilter::one_to_one>)::get().get_class_by_name("regexp");

	PVCore::PVArgumentList args;
	args["regexp"] = QString("([0-9]+)[0-9.]*\\s+[0-9]+\\s+[0-9]+\\s+[A-Z/_-]+([0-9]+)\\s+[0-9]+\\s+(GET|POST|PUT|OPTIONS)\\s+(\\S+)\\s+(\\S+)\\s+([^/]+)/(\\d+.\\d+.\\d+.\\d+)");
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
	PVFilter::PVFieldsMappingFilter::list_indexes indx;
	PVFilter::PVFieldsMappingFilter::map_filters mf;
	indx.push_back(3);
	mf[indx] = url_lib_p->f();
	PVFilter::PVFieldsMappingFilter mapping_url(mf);

	// Mapping filter for the grep filter
	indx.clear();
	mf.clear();
	indx.push_back(4);
	mf[indx] = grep_lib_p->f();
	PVFilter::PVFieldsMappingFilter mapping_grep(mf);

	// Mapping filter for the duplicate filter on the last axis after our regexp
	indx.clear();
	mf.clear();
	indx.push_back(6);
	mf[indx] = duplicate_lib_p->f();
	PVFilter::PVFieldsMappingFilter mapping_duplicate(mf);

	// Final composition
	PVFilter::PVFieldsBaseFilter_f f_final = boost::bind(mapping_grep.f(), boost::bind(mapping_url.f(), boost::bind(mapping_duplicate.f(), boost::bind(regexp_lib_p->f(), _1))));

	PVFilter::PVElementFilterByFields* elt_f = new PVFilter::PVElementFilterByFields(f_final);
	PVFilter::PVChunkFilterByElt* chk_flt = new PVFilter::PVChunkFilterByElt(elt_f->f());
	auto flt_f = chk_flt->f();

	auto res = ts.run_normalization(flt_f);
	std::string output_file = std::get<2>(res);
	size_t nelts_org = std::get<0>(res);
	size_t nelts_valid = std::get<1>(res);

	PV_VALID(nelts_valid, 761UL * nb_dup);
	PV_VALID(nelts_org, 1000UL * nb_dup);

#ifndef INSPECTOR_BENCH
	// Check output is the same as the reference
	std::cout << std::endl << output_file << " - " << ref_file << std::endl;
	PV_ASSERT_VALID(PVRush::PVUtils::files_have_same_content(output_file, ref_file));
#endif
}
