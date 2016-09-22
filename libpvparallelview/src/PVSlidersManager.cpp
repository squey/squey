/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvparallelview/PVSlidersManager.h>

/*****************************************************************************
 * PVParallelView::PVSlidersManager::new_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::new_selection_sliders(PVCol nraw_col,
                                                             const id_t id,
                                                             const int64_t y_min,
                                                             const int64_t y_max)
{
	new_range_sliders(_selection_geometries, nraw_col, id, y_min, y_max);
	_new_selection_sliders.emit(nraw_col, id, y_min, y_max);
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::new_zoom_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::new_zoom_sliders(PVCol nraw_col,
                                                        const id_t id,
                                                        const int64_t y_min,
                                                        const int64_t y_max)
{
	new_range_sliders(_zoom_geometries, nraw_col, id, y_min, y_max);
	_new_zoom_sliders.emit(nraw_col, id, y_min, y_max);
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::new_zoomedselection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::new_zoomed_selection_sliders(PVCol nraw_col,
                                                                    const id_t id,
                                                                    const int64_t y_min,
                                                                    const int64_t y_max)
{
	new_range_sliders(_zoomed_selection_geometries, nraw_col, id, y_min, y_max);
	_new_zoomed_selection_sliders.emit(nraw_col, id, y_min, y_max);
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::del_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::del_selection_sliders(PVCol nraw_col, const id_t id)
{
	_del_selection_sliders.emit(nraw_col, id);
	del_range_sliders(_selection_geometries, nraw_col, id);
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::del_zoom_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::del_zoom_sliders(PVCol nraw_col, const id_t id)
{
	_del_zoom_sliders.emit(nraw_col, id);
	del_range_sliders(_zoom_geometries, nraw_col, id);
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::del_zoomed_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::del_zoomed_selection_sliders(PVCol nraw_col, const id_t id)
{
	_del_zoomed_selection_sliders.emit(nraw_col, id);
	del_range_sliders(_zoomed_selection_geometries, nraw_col, id);
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::update_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::update_selection_sliders(PVCol nraw_col,
                                                                const id_t id,
                                                                const int64_t y_min,
                                                                const int64_t y_max)
{
	update_range_sliders(_selection_geometries, nraw_col, id, y_min, y_max);
	_update_selection_sliders.emit(nraw_col, id, y_min, y_max);
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::update_zoom_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::update_zoom_sliders(PVCol nraw_col,
                                                           const id_t id,
                                                           const int64_t y_min,
                                                           const int64_t y_max,
                                                           const ZoomSliderChange change)
{
	update_range_sliders(_zoom_geometries, nraw_col, id, y_min, y_max);
	_update_zoom_sliders.emit(nraw_col, id, y_min, y_max, change);
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::update_zoomed_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::update_zoomed_selection_sliders(PVCol nraw_col,
                                                                       const id_t id,
                                                                       const int64_t y_min,
                                                                       const int64_t y_max)
{
	update_range_sliders(_zoomed_selection_geometries, nraw_col, id, y_min, y_max);
	_update_zoomed_selection_sliders.emit(nraw_col, id, y_min, y_max);
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::iterate_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::iterate_selection_sliders(
    const range_functor_t& functor) const
{
	iterate_range_sliders(_selection_geometries, functor);
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::iterate_zoom_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::iterate_zoom_sliders(const range_functor_t& functor) const
{
	iterate_range_sliders(_zoom_geometries, functor);
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::iterate_zoomed_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::iterate_zoomed_selection_sliders(
    const range_functor_t& functor) const
{
	iterate_range_sliders(_zoomed_selection_geometries, functor);
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::new_range_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::new_range_sliders(range_geometry_set_t& range,
                                                         PVCol nraw_col,
                                                         const id_t id,
                                                         const int64_t y_min,
                                                         const int64_t y_max)
{
	update_range_sliders(range, nraw_col, id, y_min, y_max);
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::del_range_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::del_range_sliders(range_geometry_set_t& range,
                                                         PVCol nraw_col,
                                                         const id_t id)
{
	auto ai = range.find(nraw_col);
	if (ai != range.end()) {
		range_geometry_list_t::const_iterator ii = ai->second.find(id);
		if (ii != ai->second.end()) {
			ai->second.erase(ii);
		}
		if (ai->second.empty()) {
			range.erase(ai);
		}
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::update_range_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::update_range_sliders(range_geometry_set_t& range,
                                                            PVCol nraw_col,
                                                            const id_t id,
                                                            const int64_t y_min,
                                                            const int64_t y_max)
{
	range_geometry_t& geom = range[nraw_col][id];
	geom.y_min = y_min;
	geom.y_max = y_max;
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::iterate_range_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::iterate_range_sliders(const range_geometry_set_t& range,
                                                             const range_functor_t& functor) const
{
	for (const auto& it : range) {
		for (auto ait = it.second.begin(); ait != it.second.end(); ++ait) {
			functor(it.first, ait->first, ait->second);
		}
	}
}
