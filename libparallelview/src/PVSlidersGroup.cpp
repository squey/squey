
#include <pvkernel/core/PVAlgorithms.h>

#include <pvparallelview/PVSlidersGroup.h>
#include <pvparallelview/PVAbstractAxisSliders.h>
#include <pvparallelview/PVSelectionAxisSliders.h>
#include <pvparallelview/PVZoomAxisSliders.h>
#include <pvparallelview/PVZoomedSelectionAxisSliders.h>
#include <pvparallelview/PVAbstractAxisSlider.h>

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
	_zssn_obs(this),
	_zsd_obs(this),
	_ssd_obs(this),
	_zssd_obs(this),
	_axis_id(axis_id),
	_axis_scale(1.0f)
{
	// does not care about children events
	// RH: this method is obsolete in Qt 4.8 and should be replaced with
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
	                                             _zssn_obs);

	PVHive::PVHive::get().register_func_observer(_sliders_manager_p,
	                                             _zsd_obs);
	PVHive::PVHive::get().register_func_observer(_sliders_manager_p,
	                                             _ssd_obs);
	PVHive::PVHive::get().register_func_observer(_sliders_manager_p,
	                                             _zssd_obs);
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::~PVSlidersGroup
 *****************************************************************************/

PVParallelView::PVSlidersGroup::~PVSlidersGroup()
{
	remove_selection_sliders();
	remove_zoom_slider();

	if (scene()) {
		scene()->removeItem(this);
	}

	if (group()) {
		group()->removeFromGroup(this);
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::remove_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::remove_selection_sliders()
{
	sas_set_t sasliders = _selection_sliders;

	for (const auto &it : sasliders) {
		del_selection_sliders(it.first);
	}

	zsas_set_t zssliders = _zoomed_selection_sliders;

	for (const auto &it : zssliders) {
		del_zoomed_selection_sliders(it.first);
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::remove_zoom_slider
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::remove_zoom_slider()
{
	zas_set_t zasliders = _zoom_sliders;

	for (const auto &it : zasliders) {
		if (it.first == (id_t)this) {
			del_zoom_sliders(it.first);
		}
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::delete_own_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::delete_own_selection_sliders()
{
	sas_set_t ssliders = _selection_sliders;

	for (const auto &it : ssliders) {
		if (it.first != it.second) {
			continue;
		}
		PVHive::call<FUNC(PVSlidersManager::del_selection_sliders)>(_sliders_manager_p,
		                                                            get_axis_id(),
		                                                            it.first);
	}

	zsas_set_t zssliders = _zoomed_selection_sliders;

	for (const auto &it : zssliders) {
		if (it.first != it.second) {
			continue;
		}
		PVHive::call<FUNC(PVSlidersManager::del_zoomed_selection_sliders)>(_sliders_manager_p,
		                                                                   get_axis_id(),
		                                                                   it.first);
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::delete_own_zoom_slider
 *****************************************************************************/


void PVParallelView::PVSlidersGroup::delete_own_zoom_slider()
{
	PVHive::call<FUNC(PVSlidersManager::del_zoom_sliders)>(_sliders_manager_p,
	                                                       get_axis_id(), this);
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::set_axis_scale
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::set_axis_scale(float s)
{
	_axis_scale = s;

	for(auto &it : _selection_sliders) {
		it.second->refresh();
	}

	for(auto &it : _zoomed_selection_sliders) {
		it.second->refresh();
	}

	for(auto &it : _zoom_sliders) {
		it.second->refresh();
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::get_selection_ranges
 *****************************************************************************/

PVParallelView::PVSlidersGroup::selection_ranges_t
PVParallelView::PVSlidersGroup::get_selection_ranges() const
{
	selection_ranges_t ranges;

	for (const auto it : _selection_sliders) {
		ranges.push_back(it.second->get_range());
	}

	for (const auto it : _zoomed_selection_sliders) {
		ranges.push_back(it.second->get_range());
	}

	return ranges;
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::add_zoom_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::add_zoom_sliders(int64_t y_min,
                                                      int64_t y_max)
{
	PVHive::call<FUNC(PVSlidersManager::new_zoom_sliders)>(_sliders_manager_p,
	                                                       get_axis_id(),
	                                                       this,
	                                                       y_min * PVAbstractAxisSlider::precision,
	                                                       y_max * PVAbstractAxisSlider::precision);
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::add_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::add_selection_sliders(int64_t y_min,
                                                           int64_t y_max)
{
	y_min *= PVAbstractAxisSlider::precision;
	y_max *= PVAbstractAxisSlider::precision;

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
 * PVParallelView::PVSlidersGroup::add_zoomed_selection_sliders
 *****************************************************************************/

PVParallelView::PVZoomedSelectionAxisSliders *
PVParallelView::PVSlidersGroup::add_zoomed_selection_sliders(int64_t y_min,
                                                             int64_t y_max)
{
	PVZoomedSelectionAxisSliders *sliders =
		new PVZoomedSelectionAxisSliders(this, _sliders_manager_p,
		                                 this);
	add_new_zoomed_selection_sliders(sliders, sliders, y_min, y_max);

	PVHive::call<FUNC(PVSlidersManager::new_zoomed_selection_sliders)>(_sliders_manager_p,
	                                                                   get_axis_id(),
	                                                                   sliders,
	                                                                   y_min, y_max);

	return sliders;
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::del_zoom_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::del_zoom_sliders(id_t id)
{
	zas_set_t::const_iterator it = _zoom_sliders.find(id);

	if (it != _zoom_sliders.end()) {
		if (scene()) {
			scene()->removeItem(it->second);
		}
		removeFromGroup(it->second);
		delete it->second;
		_zoom_sliders.erase(it);
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::del_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::del_selection_sliders(id_t id)
{

	sas_set_t::const_iterator it = _selection_sliders.find(id);

	if (it != _selection_sliders.end()) {
		if (scene()) {
			scene()->removeItem(it->second);
		}
		removeFromGroup(it->second);
		delete it->second;
		_selection_sliders.erase(it);
	}

}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::del_zoomed_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::del_zoomed_selection_sliders(id_t id)
{
	zsas_set_t::const_iterator it = _zoomed_selection_sliders.find(id);

	if (it != _zoomed_selection_sliders.end()) {
		if (scene()) {
			scene()->removeItem(it->second);
		}
		removeFromGroup(it->second);
		delete it->second;
		_zoomed_selection_sliders.erase(it);
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::sliders_moving
 *****************************************************************************/

bool PVParallelView::PVSlidersGroup::sliders_moving() const
{
	for (const auto &it : _selection_sliders) {
		if (it.second->is_moving()) {
			return true;
		}
	}
	for (const auto &it : _zoomed_selection_sliders) {
		if (it.second->is_moving()) {
			return true;
		}
	}
	for (const auto &it : _zoom_sliders) {
		if (it.second->is_moving()) {
			return true;
		}
	}
	return false;
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::new_zoom_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::add_new_zoom_sliders(id_t id,
                                                          int64_t y_min,
                                                          int64_t y_max)
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

	_zoom_sliders[id] = sliders;
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::add_new_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::add_new_selection_sliders(PVParallelView::PVSelectionAxisSliders* sliders,
                                                               id_t id,
                                                               int64_t y_min,
                                                               int64_t y_max)
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

	_selection_sliders[id] = sliders;
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::add_new_zoomed_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::add_new_zoomed_selection_sliders(PVParallelView::PVZoomedSelectionAxisSliders* sliders,
                                                                      id_t id,
                                                                      int64_t y_min,
                                                                      int64_t y_max)
{
	if (sliders == nullptr) {
		sliders = new PVZoomedSelectionAxisSliders(this, _sliders_manager_p,
		                                           this);
	}

	if (id == nullptr) {
		id = sliders;
	}

	sliders->initialize(id, y_min, y_max);

	addToGroup(sliders);

	sliders->setPos(0, 0);

	connect(sliders, SIGNAL(sliders_moved()), this, SLOT(selection_slider_moved()));

	_zoomed_selection_sliders[id] = sliders;
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
			int64_t y_min = std::get<2>(args);
			int64_t y_max = std::get<3>(args);
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

		if (_parent->_selection_sliders.find(id) == _parent->_selection_sliders.end()) {
			int64_t y_min = std::get<2>(args);
			int64_t y_max = std::get<3>(args);
			_parent->add_new_selection_sliders(nullptr, id,
			                                   y_min, y_max);
		}
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::zoomed_selection_sliders_new_obs::update
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::zoomed_selection_sliders_new_obs::update(arguments_deep_copy_type const& args) const
{
	const axis_id_t &axis_id = std::get<0>(args);

	if (axis_id == _parent->get_axis_id()) {
		PVSlidersManager::id_t id = std::get<1>(args);

		if (_parent->_zoomed_selection_sliders.find(id) == _parent->_zoomed_selection_sliders.end()) {
			int64_t y_min = std::get<2>(args);
			int64_t y_max = std::get<3>(args);
			_parent->add_new_zoomed_selection_sliders(nullptr, id,
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
		_parent->del_zoom_sliders(std::get<1>(args));
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::selection_sliders_del_obs::update
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::selection_sliders_del_obs::update(arguments_deep_copy_type const& args) const
{
	const axis_id_t &axis_id = std::get<0>(args);

	if (axis_id == _parent->get_axis_id()) {
		_parent->del_selection_sliders(std::get<1>(args));
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::zoomed_selection_sliders_del_obs::update
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::zoomed_selection_sliders_del_obs::update(arguments_deep_copy_type const& args) const
{
	const axis_id_t &axis_id = std::get<0>(args);

	if (axis_id == _parent->get_axis_id()) {
		_parent->del_zoomed_selection_sliders(std::get<1>(args));
	}
}
