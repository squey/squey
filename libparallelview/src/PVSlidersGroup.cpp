
#include <pvkernel/core/PVAlgorithms.h>

#include <pvparallelview/PVSlidersGroup.h>

#include <QGraphicsScene>

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::PVSlidersGroup
 *****************************************************************************/

PVParallelView::PVSlidersGroup::PVSlidersGroup(PVSlidersManager_p sm_p,
                                               const axis_id_t &axis_id,
                                               QGraphicsItem *parent) :
	QGraphicsItemGroup(parent),
	_sliders_manager_p(sm_p),
	_zsn_obs(this),
	_ssn_obs(this),
	_zsd_obs(this),
	_ssd_obs(this),
	_axis_id(axis_id),
	_axis_scale(1.0f)
{
	// does not care about children events
	// RH: this method is obsolete in Qt 4.8 and replaced with
	//     setFiltersChildEvents() but this latter does not have the same
	//     behaviour... so I keep setHandlesChildEvents()
	setHandlesChildEvents(false);

	_sliders_manager_p->iterate_zoom_sliders([&](const axis_id_t &axis_id,
	                                             const id_t id,
	                                             const range_geometry_t &geom)
	                                         {
		                                         if (axis_id != get_axis_id()) {
			                                         return;
		                                         }

		                                         add_new_zoom_sliders(id,
		                                                              geom.y_min,
		                                                              geom.y_max);
	                                         });

	_sliders_manager_p->iterate_selection_sliders([&](const axis_id_t &axis_id,
	                                                  const id_t id,
	                                                  const range_geometry_t &geom)
	                                         {
		                                         if (axis_id != get_axis_id()) {
			                                         return;
		                                         }

		                                         add_new_selection_sliders(nullptr,
		                                                                   id,
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

PVParallelView::PVSlidersGroup::~PVSlidersGroup()
{
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::recreate_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::recreate_sliders()
{
	/* FIXME: all child must be removed and recreated by iterating on
	 * all sliders types.
	 */
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::set_axis_scale
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::set_axis_scale(float s)
{
	_axis_scale = s;

	for(PVAbstractAxisSliders *aas : _all_sliders) {
		aas->refresh();
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::add_zoom_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::add_zoom_sliders(uint32_t y_min,
                                                      uint32_t y_max)
{
	PVHive::call<FUNC(PVSlidersManager::new_zoom_sliders)>(_sliders_manager_p,
	                                                       get_axis_id(),
	                                                       this, y_min, y_max);
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::add_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::add_selection_sliders(uint32_t y_min,
                                                           uint32_t y_max)
{
	PVParallelView::PVSelectionAxisSliders *sliders =
		new PVParallelView::PVSelectionAxisSliders(this, _sliders_manager_p,
		                                           this);
	add_new_selection_sliders(sliders, sliders, y_min, y_max);

	PVHive::call<FUNC(PVSlidersManager::new_selection_sliders)>(_sliders_manager_p,
	                                                            get_axis_id(),
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

void PVParallelView::PVSlidersGroup::add_new_zoom_sliders(id_t id,
                                                          uint32_t y_min,
                                                          uint32_t y_max)
{
	PVParallelView::PVZoomAxisSliders* sliders =
		new PVParallelView::PVZoomAxisSliders(this, _sliders_manager_p,
		                                      this);

	if (id == nullptr) {
		id = this;
	}

	sliders->initialize(id, y_min, y_max);

	addToGroup(sliders);

	sliders->setPos(0, 0);

	_all_sliders.insert(sliders);
	_registered_ids.insert(id);
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::add_new_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::add_new_selection_sliders(PVParallelView::PVSelectionAxisSliders* sliders,
                                                               id_t id,
                                                               uint32_t y_min,
                                                               uint32_t y_max)
{
	if (sliders == nullptr) {
		sliders = new PVParallelView::PVSelectionAxisSliders(this, _sliders_manager_p,
		                                                     this);
	}

	if (id == nullptr) {
		id = sliders;
	}

	sliders->initialize(id, y_min, y_max);

	addToGroup(sliders);

	sliders->setPos(0, 0);

	connect(sliders, SIGNAL(sliders_moved()), this, SLOT(selection_slider_moved()));

	_all_sliders.insert(sliders);
	_selection_sliders.insert(sliders);
	_registered_ids.insert(id);
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::zoom_sliders_new_obs::update
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::zoom_sliders_new_obs::update(arguments_deep_copy_type const& args) const
{
	const axis_id_t &axis_id = std::get<0>(args);

	if (axis_id == _parent->get_axis_id()) {
		PVSlidersManager::id_t id = std::get<1>(args);

		if (id != _parent) {
			uint32_t y_min = std::get<2>(args);
			uint32_t y_max = std::get<3>(args);
			_parent->add_new_zoom_sliders(id, y_min, y_max);
		}
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::selection_sliders_new_obs::update
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::selection_sliders_new_obs::update(arguments_deep_copy_type const& args) const
{
	const axis_id_t &axis_id = std::get<0>(args);

	if (axis_id == _parent->get_axis_id()) {
		PVSlidersManager::id_t id = std::get<1>(args);

		if (_parent->_registered_ids.find(id) == _parent->_registered_ids.end()) {
			uint32_t y_min = std::get<2>(args);
			uint32_t y_max = std::get<3>(args);
			_parent->add_new_selection_sliders(nullptr, id,
			                                   y_min, y_max);
		}
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::zoom_sliders_del_obs::update
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::zoom_sliders_del_obs::update(arguments_deep_copy_type const& args) const
{
	const axis_id_t &axis_id = std::get<0>(args);

	if (axis_id == _parent->get_axis_id()) {
		PVSlidersManager::id_t id = std::get<1>(args);

		if (_parent->_registered_ids.find(id) != _parent->_registered_ids.end()) {
			_parent->_registered_ids.erase(id);
			_parent->_all_sliders.erase((PVParallelView::PVAbstractAxisSliders*)id);
		}
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::selection_sliders_del_obs::update
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::selection_sliders_del_obs::update(arguments_deep_copy_type const& args) const
{
	const axis_id_t &axis_id = std::get<0>(args);

	if (axis_id == _parent->get_axis_id()) {
		PVSlidersManager::id_t id = std::get<1>(args);

		if (_parent->_registered_ids.find(id) != _parent->_registered_ids.end()) {
			_parent->_registered_ids.erase(id);
			_parent->_selection_sliders.erase((PVParallelView::PVSelectionAxisSliders*)id);
			_parent->_all_sliders.erase((PVParallelView::PVAbstractAxisSliders*)id);
		}
	}
}
