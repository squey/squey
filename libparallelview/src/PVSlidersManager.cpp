
#include <pvparallelview/PVSlidersManager.h>

#include <iostream>

/*****************************************************************************
 * PVParallelView::PVSlidersManager::PVSlidersManager
 *****************************************************************************/

PVParallelView::PVSlidersManager::PVSlidersManager()
{}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::~PVSlidersManager
 *****************************************************************************/

PVParallelView::PVSlidersManager::~PVSlidersManager()
{}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::new_zoom_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::new_zoom_sliders(const PVZoneID axis,
                                                        const id_t id,
                                                        const uint32_t y_min,
                                                        const uint32_t y_max)
{
	new_interval_sliders(_zoom_geometries, axis, id, y_min, y_max);
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::new_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::new_selection_sliders(const PVZoneID axis,
                                                             const id_t id,
                                                             const uint32_t y_min,
                                                             const uint32_t y_max)
{
	new_interval_sliders(_selection_geometries, axis, id, y_min, y_max);
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::del_zoom_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::del_zoom_sliders(const PVZoneID axis,
                                                        const id_t id)
{
	del_interval_sliders(_zoom_geometries, axis, id);
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::del_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::del_selection_sliders(const PVZoneID axis,
                                                             const id_t id)
{
	del_interval_sliders(_selection_geometries, axis, id);
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::update_zoom_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::update_zoom_sliders(const PVZoneID axis,
                                                           const id_t id,
                                                           const uint32_t y_min,
                                                           const uint32_t y_max,
                                                           const ZoomSliderChange)
{
	// the last parameter is useless for the manager
	update_interval_sliders(_zoom_geometries, axis, id, y_min, y_max);
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::update_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::update_selection_sliders(const PVZoneID axis,
                                                                const id_t id,
                                                                const uint32_t y_min,
                                                                const uint32_t y_max)
{
	update_interval_sliders(_selection_geometries, axis, id, y_min, y_max);
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::iterate_zoom_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::iterate_zoom_sliders(const interval_functor_t &functor) const
{
	iterate_interval_sliders(_zoom_geometries, functor);
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::iterate_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::iterate_selection_sliders(const interval_functor_t &functor) const
{
	iterate_interval_sliders(_selection_geometries, functor);
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::new_interval_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::new_interval_sliders(interval_geometry_set_t &interval,
                                                            const PVZoneID axis,
                                                            const id_t id,
                                                            const uint32_t y_min,
                                                            const uint32_t y_max)
{
	update_interval_sliders(interval, axis, id, y_min, y_max);
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::del_interval_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::del_interval_sliders(interval_geometry_set_t &interval,
                                                            const PVZoneID axis,
                                                            const id_t id)
{
	interval_geometry_set_t::iterator ai = interval.find(axis);
	if (ai != interval.end()) {
		interval_geometry_list_t::const_iterator ii = ai->second.find(id);
		if (ii != ai->second.end()) {
			ai->second.erase(ii);
		}
		if (ai->second.empty()) {
			interval.erase(ai);
		}
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::update_interval_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::update_interval_sliders(interval_geometry_set_t &interval,
                                                               const PVZoneID axis,
                                                               const id_t id,
                                                               const uint32_t y_min,
                                                               const uint32_t y_max)
{
	interval_geometry_t &geom = interval[axis][id];
	geom.y_min = y_min;
	geom.y_max = y_max;
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::iterate_interval_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::iterate_interval_sliders(const interval_geometry_set_t &interval,
                                                                const interval_functor_t &functor) const
{
	for(interval_geometry_set_t::const_iterator it = interval.begin();
	    it != interval.end(); ++it) {
		for(interval_geometry_list_t::const_iterator ait = it->second.begin();
		    ait != it->second.end(); ++ait) {
			functor(it->first, ait->first, ait->second);
		}
	}
}
