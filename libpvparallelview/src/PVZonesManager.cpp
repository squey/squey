/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVHardwareConcurrency.h>
#include <inendi/PVView.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVZoneProcessing.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/task_scheduler_init.h>

#include <QSet>

namespace PVParallelView
{
namespace __impl
{

class ZoneCreation
{
  public:
	void operator()(const tbb::blocked_range<PVZoneID>& r) const
	{
		PVParallelView::PVZonesManager* zm = _zm;
		PVParallelView::PVZoneTree::ProcessData pdata;
		for (PVZoneID z = r.begin(); z != r.end(); z++) {
			pdata.clear();
			PVZoneTree& ztree = zm->_zones[z].ztree();
			ztree.process(zm->get_zone_processing(z), pdata);
		}
	}

  public:
	PVParallelView::PVZonesManager* _zm;
};
} // namespace __impl
} // namespace PVParallelView

/******************************************************************************
 *
 * PVParallelView::PVZonesManager::PVZonesManager
 *
 *****************************************************************************/
PVParallelView::PVZonesManager::PVZonesManager(Inendi::PVView const& view)
    : _plotted(view.get_parent<Inendi::PVPlotted>())
    , _nrows(view.get_row_count())
    , _ncols(view.get_column_count())
    , _axes_comb(view.get_axes_combination().get_combination())
{
	update_all();
}

/******************************************************************************
 *
 * PVParallelView::PVZonesManager::update_all
 *
 *****************************************************************************/
void PVParallelView::PVZonesManager::update_all()
{
	PVZoneID nzones = get_number_of_managed_zones();
	PVLOG_INFO("(PVZonesManager::update_all) number of zones = %d\n", nzones);
	assert(nzones >= 1);
	_zones.clear();
	_zones.resize(nzones);

	__impl::ZoneCreation zc;
	zc._zm = this;
	const size_t nthreads = PVCore::PVHardwareConcurrency::get_physical_core_number();
	tbb::task_scheduler_init init(nthreads);
	tbb::parallel_for(tbb::blocked_range<PVZoneID>(0, nzones, 8), zc);
}

/******************************************************************************
 *
 * PVParallelView::PVZonesManager::update_zone
 *
 *****************************************************************************/
void PVParallelView::PVZonesManager::update_zone(PVZoneID zone_id)
{
	assert(zone_id < (PVZoneID)_zones.size());

	_zones[zone_id] = PVZone();

	PVZoneProcessing zp = get_zone_processing(zone_id);
	PVParallelView::PVZoneTree::ProcessData pdata;
	pdata.clear();

	PVZoneTree& ztree = _zones[zone_id].ztree();
	ztree.process(zp, pdata);

	PVZoomedZoneTree& zztree = _zones[zone_id].zoomed_ztree();
	zztree.reset();
}

/******************************************************************************
 *
 * PVParallelView::PVZonesManager::update_from_axes_comb
 *
 *****************************************************************************/
std::vector<PVZoneID>
PVParallelView::PVZonesManager::update_from_axes_comb(std::vector<PVCol> const& ac)
{
	typedef std::pair<PVCol, PVCol> axes_pair_t;
	typedef std::vector<axes_pair_t> axes_pair_list_t;

	/* the principle is to contruct a new PVZone list by collecting
	 * unchanged PVZone from the old PVZone list. So that, only the new
	 * entries of the new list are updated.
	 */

	/* to help finding unchanged zones, we use a list of axis index pair
	 * to identify them.
	 */
	axes_pair_list_t old_pairs;
	PVCol old_nb_pairs = _axes_comb.size() - 1;
	old_pairs.reserve(old_nb_pairs);

	for (PVCol i = 0; i < old_nb_pairs; ++i) {
		old_pairs.push_back(std::make_pair(_axes_comb[i], _axes_comb[i + 1]));
	}

	std::vector<PVZoneID> zoneids;
	PVCol new_nb_pairs = ac.size() - 1;
	std::vector<PVZone> new_zones(new_nb_pairs);

	// iterate on the new axes combination to find reusable zones
	for (PVCol i = 0; i < new_nb_pairs; i++) {
		axes_pair_t axes_pair = std::make_pair(ac[i], ac[i + 1]);
		axes_pair_list_t::iterator it = std::find(old_pairs.begin(), old_pairs.end(), axes_pair);

		if (it == old_pairs.end()) {
			// this zone has to be updated (when _zone will be updated)
			zoneids.push_back(i);
		} else {
			// this zone is unchanged, copying it.
			new_zones[i] = _zones[it - old_pairs.begin()];
		}
	}

	_zones = new_zones;
	_axes_comb = ac;

	// finally, the new zones are updated
	for (PVZoneID zone_id : zoneids) {
		update_zone(zone_id);
	}

	return zoneids;
}

/******************************************************************************
 *
 * PVParallelView::PVZonesManager::update_from_axes_comb
 *
 *****************************************************************************/
std::vector<PVZoneID>
PVParallelView::PVZonesManager::update_from_axes_comb(Inendi::PVView const& view)
{
	return update_from_axes_comb(view.get_axes_combination().get_combination());
}

/******************************************************************************
 *
 * PVParallelView::PVZonesManager::request_zoomed_zone
 *
 *****************************************************************************/
void PVParallelView::PVZonesManager::request_zoomed_zone(PVZoneID zone_id)
{
	const size_t nthreads = PVCore::PVHardwareConcurrency::get_physical_core_number();
	tbb::task_scheduler_init init(nthreads);

	PVZoomedZoneTree& zztree = _zones[zone_id].zoomed_ztree();

	if (zztree.is_initialized()) {
		return;
	}

	BENCH_START(zztree);
	PVZoneProcessing zp = get_zone_processing(zone_id);
	PVZoneTree& ztree = _zones[zone_id].ztree();
	zztree.process(zp, ztree);
	BENCH_END(zztree, "ZZTREES PROCESS", 1, 1, 1, 1);
}

/******************************************************************************
 *
 * PVParallelView::PVZonesManager::filter_zone_by_sel
 *
 *****************************************************************************/
void PVParallelView::PVZonesManager::filter_zone_by_sel(PVZoneID zone_id,
                                                        const Inendi::PVSelection& sel)
{
	assert(zone_id < (PVZoneID)_zones.size());
	_zones[zone_id].filter_by_sel(sel);
}

void PVParallelView::PVZonesManager::filter_zone_by_sel_background(PVZoneID zone_id,
                                                                   const Inendi::PVSelection& sel)
{
	assert(zone_id < (PVZoneID)_zones.size());
	_zones[zone_id].filter_by_sel_background(sel);
}

/******************************************************************************
 *
 * PVParallelView::PVZonesManager::list_cols_to_zones
 *
 *****************************************************************************/
QSet<PVZoneID> PVParallelView::PVZonesManager::list_cols_to_zones(QSet<PVCol> const& cols) const
{
	QSet<PVZoneID> ret;
	for (PVCol c : cols) {
		if (c == 0) {
			ret << 0;
		} else if (c == get_number_of_managed_zones()) {
			ret << c - 1;
		} else {
			ret << c;
			ret << c - 1;
		}
	}
	return ret;
}
