/**
 * \file PVZoneTreeNoAlloc.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVZONETREENOALLOC_H
#define PVPARALLELVIEW_PVZONETREENOALLOC_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVPODTree.h>

#include <pvparallelview/PVZoneTreeBase.h>

#include <tbb/enumerable_thread_specific.h>

namespace PVParallelView {

class PVZoneProcessing;

class PVZoneTreeNoAlloc: public PVZoneTreeBase
{
public:
	typedef PVCore::PVPODTree<uint32_t, uint32_t, NBUCKETS> Tree;
	typedef std::vector<PVRow, tbb::scalable_allocator<PVRow> > vect;
	typedef tbb::enumerable_thread_specific<vect> TLS;
public:
	PVZoneTreeNoAlloc():
		PVZoneTreeBase()
	{ }
public:
	void process_sse(PVZoneProcessing const& zp);
	void process_omp_sse(PVZoneProcessing const& zp);

	void filter_by_sel_omp(Picviz::PVSelection const& sel);
	void filter_by_sel_tbb(Picviz::PVSelection const& sel);

	size_t browse_tree_bci_by_sel(PVHSVColor* colors, PVBCICode<NBITS_INDEX>* codes, Picviz::PVSelection const& sel);

private:
	void get_float_pts(pts_t& pts, Picviz::PVPlotted::plotted_table_t const& org_plotted, PVRow nrows, PVCol col_a, PVCol col_b);

public://private:
	Tree _tree;
};

}

#endif
