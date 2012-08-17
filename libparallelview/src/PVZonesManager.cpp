/**
 * \file PVZonesManager.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVHardwareConcurrency.h>
#include <picviz/PVView.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVZoneProcessing.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/task_scheduler_init.h>

namespace PVParallelView { namespace __impl {

class ZoneCreation
{
public:
	void operator()(const tbb::blocked_range<PVZoneID>& r) const
	{
		PVParallelView::PVZonesManager* zm = _zm;
		PVParallelView::PVZoneProcessing zp(zm->get_uint_plotted(), zm->get_number_rows());
		PVParallelView::PVZoneTree::ProcessData &pdata = _tls_pdata.local();
		for (PVZoneID z = r.begin(); z != r.end(); z++) {
			pdata.clear();
			zm->get_zone_cols(z, zp.col_a(), zp.col_b());
			PVZoneTree& ztree = zm->_zones[z].ztree();
			ztree.process(zp, pdata);
		}
	}

public:
	PVParallelView::PVZonesManager* _zm;

private:
    mutable tbb::enumerable_thread_specific<PVParallelView::PVZoneTree::ProcessData> _tls_pdata;
};

} }

PVParallelView::PVZonesManager::PVZonesManager()
{
}

void PVParallelView::PVZonesManager::set_uint_plotted(Picviz::PVPlotted::uint_plotted_table_t const& plotted, PVRow nrows, PVCol ncols)
{
	_uint_plotted = &plotted;
	_nrows = nrows;
	_ncols = ncols;

	// Init original axes combination
	_axes_comb.clear();
	_axes_comb.reserve(ncols);
	for (PVCol c = 0; c < ncols; c++) {
		_axes_comb.push_back(c);
	}
}

void PVParallelView::PVZonesManager::update_all()
{
	PVZoneID nzones = get_number_zones();
	PVLOG_INFO("(PVZonesManager::update_all) number of zones = %d\n", nzones);
	assert(nzones >= 1);
	_zones.clear();
	_zones.reserve(nzones);
	for (PVZoneID z = 0; z < nzones; z++) {
		_zones.push_back(PVZone());
	}
	
	PVZoneProcessing zp(get_uint_plotted(), get_number_rows());
	{
		__impl::ZoneCreation zc;
		zc._zm = this;
		const size_t nthreads = PVCore::PVHardwareConcurrency::get_physical_core_number();
		tbb::task_scheduler_init init(nthreads);
		tbb::parallel_for(tbb::blocked_range<PVZoneID>(0, nzones, 8), zc);


		BENCH_START(zztree);
#if 1
		// Create Zoomed Zone Tree (serial)
		for (PVZoneID z = 0; z < nzones; z++) {
			get_zone_cols(z, zp.col_a(), zp.col_b());
			PVZoneTree& ztree = _zones[z].ztree();
			PVZoomedZoneTree& zztree = _zones[z].zoomed_ztree();
			zztree.process(zp, ztree);
		}
		BENCH_END(zztree, "ZZTREES PROCESS (SERIAL)", 1, 1, 1, 1);
#else
		// Create Zoomed Zone Tree (parallel)
		tbb::parallel_for(tbb::blocked_range<size_t>(0, nzones, 1), [&](tbb::blocked_range<size_t> const& range) {
			for (size_t z = range.begin(); z != range.end(); z++) {
				get_zone_cols(z, zp.col_a(), zp.col_b());
				PVZoneTree& ztree = _zones[z].ztree();
				PVZoomedZoneTree& zztree = _zones[z].zoomed_ztree();
				zztree.process(zp, ztree);
			}
		});
		BENCH_END(zztree, "ZZTREES PROCESS (PARALLEL)", 1, 1, 1, 1);
#endif

	}

	/*
	tbb::task_scheduler_init init(atol(getenv("NUM_THREADS")));
	PVParallelView::PVZoneTree::ProcessTLS tls;
	for (PVZoneID z = 0; z < nzones; z++) {
		get_zone_cols(z, zp.col_a(), zp.col_b());
		PVZoneTree& ztree = _zones[z].ztree();
		ztree.process(zp, tls);
		//PVZoomedZoneTree& zztree = _zones[z].zoomed_ztree();
		//zztree.process(zp, ztree);
	}*/
}

void PVParallelView::PVZonesManager::update_from_axes_comb(QVector<PVCol> const& ac)
{
	// TODO: optimise this to update only the concerned zones !
	_axes_comb = ac;
	update_all();
}

void PVParallelView::PVZonesManager::update_from_axes_comb(Picviz::PVView const& view)
{
	update_from_axes_comb(view.get_axes_combination().get_axes_index_list());
}

void PVParallelView::PVZonesManager::filter_zone_by_sel(PVZoneID zid, const Picviz::PVSelection& sel)
{
	assert(zid < (PVZoneID) _zones.size());

	PVParallelView::PVZoneProcessing zp(get_uint_plotted(), get_number_rows(), zid, zid+1);
	_zones[zid].ztree().filter_by_sel(sel);
}

void PVParallelView::PVZonesManager::invalidate_selection()
{
	for (PVZone& zone : _zones) {
		zone.invalid_selection();
	}
}

void PVParallelView::PVZonesManager::set_uint_plotted(Picviz::PVView const& view)
{
	set_uint_plotted(view.get_parent<Picviz::PVPlotted>()->get_uint_plotted(), view.get_row_count(), view.get_column_count());
}

uint32_t PVParallelView::PVZonesManager::get_zone_absolute_pos(PVZoneID zone) const
{
	assert(zone < (PVZoneID) _zones.size());
	uint32_t pos = 0;
	for (PVZoneID z = 0; z < zone; z++) {
		pos += _zones[z].width() + PVParallelView::AxisWidth;
	}
	return pos;
}

PVZoneID PVParallelView::PVZonesManager::get_zone_id(int abs_pos) const
{
	PVZoneID zid = 0;
	ssize_t pos = 0;
	for (; zid < (PVZoneID) (_zones.size()-1) ; zid++)
	{
		pos += _zones[zid].width() + PVParallelView::AxisWidth;
		if (pos > abs_pos) {
			break;
		}
	}

	assert(zid < (PVZoneID) _zones.size());
	return zid;
}
