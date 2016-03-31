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
constexpr static size_t nb_dup = 20;
#else
constexpr static size_t nb_dup = 1;
#endif

static constexpr const char* log_file = TEST_FOLDER "/pvkernel/rush/splitters/csv/proxy_sample.csv";
static constexpr const char* ref_file = TEST_FOLDER "/pvkernel/rush/splitters/csv/proxy_sample.csv.out";


int main()
{
	pvtest::TestSplitter ts(log_file, nb_dup);

	// Prepare splitter plugin
	PVFilter::PVFieldsSplitter::p_type sp_lib_p = LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_class_by_name("csv");

	PVCore::PVArgumentList args = sp_lib_p->get_args();
	args["sep"] = QVariant(QChar(','));
	sp_lib_p->set_args(args);

	PVFilter::PVElementFilterByFields* elt_f = new PVFilter::PVElementFilterByFields(sp_lib_p->f());
	PVFilter::PVChunkFilterByElt* chk_flt = new PVFilter::PVChunkFilterByElt(elt_f->f());
	PVFilter::PVChunkFilter_f flt_f = chk_flt->f();

	std::string output_file = pvtest::get_tmp_filename();
	// Extract source and split fields.
	{
		std::ofstream ofs(output_file);

		size_t nelts_org = 0;
		size_t nelts_valid = 0;
		std::chrono::duration<double> dur;
		decltype(std::chrono::steady_clock::now()) start;
		// TODO : Add parallelism on Chunk splitter!!
		while (PVCore::PVChunk* pc = ts.get_source()()) {
			start = std::chrono::steady_clock::now();
			flt_f(pc);
			dur += std::chrono::steady_clock::now() - start;
			size_t no = 0;
			size_t nv = 0;
			pc->get_elts_stat(no, nv);
			nelts_org += no;
			nelts_valid += nv;
			dump_chunk_csv(*pc, ofs);
			pc->free();
		}
		std::cout << dur.count();

		PV_VALID(nelts_valid, nelts_org);
	}

#ifndef INSPECTOR_BENCH
	// Check output is the same as the reference
	PV_ASSERT_VALID(PVRush::PVUtils::files_have_same_content(output_file, ref_file));
#endif
	return 0;
}
