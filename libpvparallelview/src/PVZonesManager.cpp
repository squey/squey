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
	/* PVZoneID must not be used because it prevents TBB to do arithmetic on
	 * blocked range
	 */
	void operator()(const tbb::blocked_range<size_t>& r) const
	{
		PVParallelView::PVZonesManager* zm = _zm;
		PVParallelView::PVZoneTree::ProcessData pdata;
		for (size_t index = r.begin(); index != r.end(); ++index) {
			pdata.clear();
			auto & [ zid, zone ] = zm->_zones[index];
			PVZoneTree& ztree = zone.ztree();
			ztree.process(zm->get_zone_processing(zid), pdata);
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
PVParallelView::PVZonesManager::PVZonesManager(Inendi::PVView const& view) : _view(view)
{
	update_from_axes_comb(_view);
}

/******************************************************************************
 *
 * PVParallelView::PVZonesManager::update_all
 *
 *****************************************************************************/
void PVParallelView::PVZonesManager::update_all(bool reinit_zones)
{
	size_t nzones = get_number_of_zones();
	PVLOG_INFO("(PVZonesManager::update_all) number of zones = %d\n", nzones);
	assert(nzones >= 1);
	if (reinit_zones) {
		for (auto& zone : _zones) {
			zone.second = PVZone();
		}
	}

	__impl::ZoneCreation zc;
	zc._zm = this;
	const size_t nthreads = PVCore::PVHardwareConcurrency::get_physical_core_number();
	tbb::task_scheduler_init init(nthreads);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, nzones, 8), zc);
}

/******************************************************************************
 *
 * PVParallelView::PVZonesManager::update_zone
 *
 *****************************************************************************/
void PVParallelView::PVZonesManager::update_zone(PVZoneID zone_id)
{
	PVZone& zone = get_zone(zone_id);

	zone = PVZone();

	PVZoneProcessing zp = get_zone_processing(zone_id);
	PVParallelView::PVZoneTree::ProcessData pdata;
	pdata.clear();

	PVZoneTree& ztree = zone.ztree();
	ztree.process(zp, pdata);

	PVZoomedZoneTree& zztree = zone.zoomed_ztree();
	zztree.reset();
}

/******************************************************************************
 *
 * PVParallelView::PVZonesManager::acquire_zone
 *
 *****************************************************************************/
auto PVParallelView::PVZonesManager::acquire_zone(PVZoneID zone_id) -> ZoneRetainer
{
	if (not _zone_indices.count(zone_id)) {
		PVLOG_DEBUG("PVZonesManager::acquire_zone(%d:%d)\n", zone_id.first, zone_id.second);
		_zone_indices.emplace(zone_id, _zones.size());
		_zones.emplace_back(zone_id, PVZone{});
		update_zone(zone_id);
	}
	return ZoneRetainer{*this, zone_id};
}

/******************************************************************************
 *
 * PVParallelView::PVZonesManager::release_zone
 *
 *****************************************************************************/
void PVParallelView::PVZonesManager::release_zone(PVZoneID zone_id)
{
	if (zone_id == PVZONEID_INVALID) {
		return;
	}

	PVLOG_DEBUG("PVZonesManager::release_zone(%d:%d)\n", zone_id.first, zone_id.second);

	if (_zones_ref_count.count(zone_id)) {
		return;
	}

	update_from_axes_comb(_axes_comb);
}

/******************************************************************************
 *
 * PVParallelView::PVZonesManager::update_from_axes_comb
 *
 *****************************************************************************/
void PVParallelView::PVZonesManager::update_from_axes_comb(std::vector<PVCol> const& ac)
{
	/* the principle is to contruct a new PVZone list by collecting
	 * unchanged PVZone from the old PVZone list. So that, only the new
	 * entries of the new list are updated.
	 */

	size_t new_managed_sized = ac.size() >= 2 ? ac.size() - 1 : 0;
	size_t old_managed_size = _axes_comb.size() - 1;
	size_t old_total_size = _zones.size();

	decltype(_zones) _new_zones;
	decltype(_zone_indices) _new_zone_indices;
	std::vector<PVZoneID> to_be_updated_new_zones;

	// Not exact but reasonnable asumption
	_new_zones.reserve(_zones.size() + new_managed_sized - old_managed_size);

	// iterate on the new axes combination to find reusable zones
	for (size_t i = 0; i < new_managed_sized; ++i) {
		PVZoneID zid{ac[i], ac[i + 1]};
		if (auto it = _zone_indices.find(zid); it != _zone_indices.end()) {
			// this zone is unchanged, copying it.
			_new_zones.push_back(_zones[it->second]);
		} else {
			// this zone has to be updated (when _zones will be updated)
			_new_zones.emplace_back(zid, PVZone{});
			to_be_updated_new_zones.push_back(zid);
		}
		_new_zone_indices.emplace(zid, i);
	}

	// zones that are still owned by ZoneRetainer are copied too
	for (auto& old_zone : _zones) {
		if (_zones_ref_count.count(old_zone.first) > 0 &&
		    not _new_zone_indices.count(old_zone.first)) {
			_new_zones.push_back(old_zone);
			_new_zone_indices.emplace(old_zone.first, _new_zones.size() - 1);
		}
	}

	_axes_comb = ac;
	std::swap(_zones, _new_zones);
	std::swap(_zone_indices, _new_zone_indices);

	// finally, the new zones are updated
	if (old_total_size == 0) {
		update_all(false);
	} else {
		for (auto& to_be_updated : to_be_updated_new_zones) {
			update_zone(to_be_updated);
		}
	}

	PVLOG_DEBUG("PVParallelView::PVZonesManager::update_from_axes_comb: %d zones (%d zones from "
	            "axes combination)\n",
	            _zones.size(), _axes_comb.size() - 1);
}

/******************************************************************************
 *
 * PVParallelView::PVZonesManager::update_from_axes_comb
 *
 *****************************************************************************/
void PVParallelView::PVZonesManager::update_from_axes_comb(Inendi::PVView const& view)
{
	update_from_axes_comb(view.get_axes_combination().get_combination());
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

	PVZoomedZoneTree& zztree = get_zoom_zone_tree(zone_id);

	if (zztree.is_initialized()) {
		return;
	}

	BENCH_START(zztree);
	PVZoneProcessing zp = get_zone_processing(zone_id);
	PVZoneTree& ztree = get_zone(zone_id).ztree();
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
	get_zone(zone_id).filter_by_sel(sel);
}

void PVParallelView::PVZonesManager::filter_zone_by_sel_background(PVZoneID zone_id,
                                                                   const Inendi::PVSelection& sel)
{
	get_zone(zone_id).filter_by_sel_background(sel);
}

/******************************************************************************
 *
 * PVParallelView::PVZonesManager::list_cols_to_zones_indices
 *
 *****************************************************************************/
std::unordered_set<PVZoneID>
PVParallelView::PVZonesManager::list_cols_to_zones_indices(QSet<PVCombCol> const& comb_cols) const
{
	std::unordered_set<PVZoneID> ret;
	for (PVCombCol comb_col : comb_cols) {
		if (comb_col == 0) {
			ret.emplace(_axes_comb[comb_col], _axes_comb[comb_col + 1]);
		} else if (comb_col == PVCombCol(get_number_of_axes_comb_zones())) {
			ret.emplace(_axes_comb[comb_col - 1], _axes_comb[comb_col]);
		} else {
			ret.emplace(_axes_comb[comb_col], _axes_comb[comb_col + 1]);
			ret.emplace(_axes_comb[comb_col - 1], _axes_comb[comb_col]);
		}
	}
	return ret;
}
