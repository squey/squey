
#include <pvkernel/core/PVAlgorithms.h>

#include <pvparallelview/PVSlidersGroup.h>
#include <pvparallelview/PVAxisSlider.h>

#include <QGraphicsScene>

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::PVSlidersGroup
 *****************************************************************************/

PVParallelView::PVSlidersGroup::PVSlidersGroup(PVSlidersManager_p sm_p,
                                               PVCol axis_index,
                                               QGraphicsItem *parent) :
	QGraphicsItemGroup(parent),
	_sliders_manager_p(sm_p),
	_zsn_obs(this),
	_ssn_obs(this),
	_zsd_obs(this),
	_ssd_obs(this),
	_axis_index(axis_index)
{
	// does not care about children events
	// RH: this method is obsolete in Qt 4.8 and replaced with
	//     setFiltersChildEvents() but this latter does not have the same
	//     behaviour... so I keep setHandlesChildEvents()
	setHandlesChildEvents(false);

	_sliders_manager_p->iterate_zoom_sliders([&](const PVCol axis,
	                                             const id_t id,
	                                             const interval_geometry_t &geom)
	                                         {
		                                         if (axis != axis_index) {
			                                         return;
		                                         }

		                                         add_new_zoom_sliders(axis, id,
		                                                              geom.y_min,
		                                                              geom.y_max);
	                                         });

	_sliders_manager_p->iterate_selection_sliders([&](const PVCol axis,
	                                                  const id_t id,
	                                                  const interval_geometry_t &geom)
	                                         {
		                                         if (axis != axis_index) {
			                                         return;
		                                         }

		                                         add_new_selection_sliders(nullptr,
		                                                                   axis, id,
		                                                                   geom.y_min,
		                                                                   geom.y_max);
	                                         });

	PVHive::PVHive::get().register_func_observer(_sliders_manager_p,
	                                             _zsn_obs);
	PVHive::PVHive::get().register_func_observer(_sliders_manager_p,
	                                             _ssn_obs);

	PVHive::PVHive::get().register_func_observer(_sliders_manager_p,
	                                             _zsd_obs);
	PVHive::PVHive::get().register_func_observer(_sliders_manager_p,
	                                             _ssd_obs);
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::add_zoom_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::add_zoom_sliders(uint32_t y_min,
                                                      uint32_t y_max)
{
	PVHive::call<FUNC(PVSlidersManager::new_zoom_sliders)>(_sliders_manager_p,
	                                                       _axis_index,
	                                                       this, y_min, y_max);
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::add_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::add_selection_sliders(uint32_t y_min,
                                                           uint32_t y_max)
{
	PVParallelView::PVSelectionAxisSliders *sliders =
		new PVParallelView::PVSelectionAxisSliders(this);
	add_new_selection_sliders(sliders, _axis_index, sliders, y_min, y_max);

	PVHive::call<FUNC(PVSlidersManager::new_selection_sliders)>(_sliders_manager_p,
	                                                            _axis_index,
	                                                            sliders,
	                                                            y_min, y_max);
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::get_selection_ranges
 *****************************************************************************/

PVParallelView::PVSlidersGroup::selection_ranges_t
PVParallelView::PVSlidersGroup::get_selection_ranges() const
{
	selection_ranges_t ranges;

	for (sas_set_t::const_iterator it = _selection_sliders.begin(); it != _selection_sliders.end(); ++it) {
		ranges.push_back((*it)->get_range());
	}

	return ranges;
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::sliders_moving
 *****************************************************************************/

bool PVParallelView::PVSlidersGroup::sliders_moving() const
{
	for (PVParallelView::PVAbstractAxisSliders* sliders : _all_sliders) {
		if (sliders->is_moving()) {
			return true;
		}
	}
	return false;
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::new_zoom_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::add_new_zoom_sliders(PVCol axis,
                                                          id_t id,
                                                          uint32_t y_min,
                                                          uint32_t y_max)
{
	PVParallelView::PVZoomAxisSliders* sliders =
		new PVParallelView::PVZoomAxisSliders(this);

	if (id == nullptr) {
		id = this;
	}

	sliders->initialize(_sliders_manager_p, axis, id, y_min, y_max);

	addToGroup(sliders);

	sliders->setPos(0, 0);

	_all_sliders.push_back(sliders);
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::add_new_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::add_new_selection_sliders(PVParallelView::PVSelectionAxisSliders* sliders,
                                                               PVCol axis,
                                                               id_t id,
                                                               uint32_t y_min,
                                                               uint32_t y_max)
{
	if (sliders == nullptr) {
		sliders = new PVParallelView::PVSelectionAxisSliders(this);
	}

	std::cout << "#### add new selection for axis: " << axis
	          << " id: " << id
	          << " min: " << y_min
	          << " max: " << y_max << std::endl;

	if (id == nullptr) {
		id = sliders;
		std::cout << "     and id: " << sliders << std::endl;
	}

	sliders->initialize(_sliders_manager_p, axis, id, y_min, y_max);

	addToGroup(sliders);

	sliders->setPos(0, 0);

	connect(sliders, SIGNAL(sliders_moved()), this, SLOT(selection_slider_moved()));

	_all_sliders.push_back(sliders);
	_selection_sliders.push_back(sliders);
	_registered_ids.insert(id);
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::zoom_sliders_new_obs::update
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::zoom_sliders_new_obs::update(arguments_deep_copy_type const& args) const
{
	PVCol axis = std::get<0>(args);

	if (axis == _parent->_axis_index) {
		PVSlidersManager::id_t id = std::get<1>(args);
		if (id != _parent) {
			uint32_t y_min = std::get<2>(args);
			uint32_t y_max = std::get<3>(args);
			_parent->add_new_zoom_sliders(axis, id, y_min, y_max);
		}
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::selection_sliders_new_obs::update
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::selection_sliders_new_obs::update(arguments_deep_copy_type const& args) const
{
	PVCol axis = std::get<0>(args);

	if (axis == _parent->_axis_index) {
		PVSlidersManager::id_t id = std::get<1>(args);

		if (_parent->_registered_ids.find(id) == _parent->_registered_ids.end()) {
			uint32_t y_min = std::get<2>(args);
			uint32_t y_max = std::get<3>(args);
			_parent->add_new_selection_sliders(nullptr, axis, id,
			                                   y_min, y_max);
		}
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::zoom_sliders_del_obs::update
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::zoom_sliders_del_obs::update(arguments_deep_copy_type const& args) const
{
	PVCol axis = std::get<0>(args);

	if (axis == _parent->_axis_index) {
		PVSlidersManager::id_t id = std::get<1>(args);

		if (_parent->_registered_ids.find(id) != _parent->_registered_ids.end()) {
			_parent->_registered_ids.erase(id);
		}
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::selection_sliders_del_obs::update
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::selection_sliders_del_obs::update(arguments_deep_copy_type const& args) const
{
	PVCol axis = std::get<0>(args);

	if (axis == _parent->_axis_index) {
		PVSlidersManager::id_t id = std::get<1>(args);

		if (_parent->_registered_ids.find(id) != _parent->_registered_ids.end()) {
			_parent->_registered_ids.erase(id);
		}
	}
}
