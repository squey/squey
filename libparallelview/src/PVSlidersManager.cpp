
#include <pvparallelview/PVSlidersManager.h>

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

void PVParallelView::PVSlidersManager::new_zoom_sliders(const PVCol axis,
                                                        const id_t id,
                                                        const uint32_t y_min,
                                                        const uint32_t y_max)
{
	update_zoom_sliders(axis, id, y_min, y_max);
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::del_zoom_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::del_zoom_sliders(const PVCol axis,
                                                        const id_t id)
{
	zoom_geometry_set_t::iterator ai = _zoom_geometries.find(axis);
	if (ai != _zoom_geometries.end()) {
		zoom_geometry_list_t::const_iterator ii = ai->second.find(id);
		if (ii != ai->second.end()) {
			ai->second.erase(ii);
		}
		if (ai->second.empty()) {
			_zoom_geometries.erase(ai);
		}
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::update_zoom_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::update_zoom_sliders(const PVCol axis,
                                                           const id_t id,
                                                           const uint32_t y_min,
                                                           const uint32_t y_max)
{
	zoom_geometry_t &geom = _zoom_geometries[axis][id];
	geom.y_min = y_min;
	geom.y_max = y_max;
}

/*****************************************************************************
 * PVParallelView::PVSlidersManager::iterate_zoom_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersManager::iterate_zoom_sliders(const zoom_functor_t &functor) const
{
	for(zoom_geometry_set_t::const_iterator it = _zoom_geometries.begin();
	    it != _zoom_geometries.end(); ++it) {
		for(zoom_geometry_list_t::const_iterator ait = it->second.begin();
		    ait != it->second.end(); ++ait) {
			functor(it->first, ait->first, ait->second);
		}
	}
}
