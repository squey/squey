
#include <pvparallelview/PVZoomAxisSliders.h>
#include <pvparallelview/PVSlidersGroup.h>
#include <pvparallelview/PVZoomAxisSlider.h>

#include <QGraphicsScene>

/*****************************************************************************
 * PVParallelView::PVZoomAxisSliders::PVZoomAxisSliders
 *****************************************************************************/

PVParallelView::PVZoomAxisSliders::PVZoomAxisSliders(QGraphicsItem *parent,
                                                     PVSlidersManager_p sm_p,
                                                     PVSlidersGroup *group) :
	PVAbstractRangeAxisSliders(parent, sm_p, group, "zoom"),
	_zsd_obs(this),
	_zsu_obs(this)
{
}

/*****************************************************************************
 * PVParallelView::PVZoomAxisSliders::initialize
 *****************************************************************************/

void PVParallelView::PVZoomAxisSliders::initialize(id_t id,
                                                   int64_t y_min,
                                                   int64_t y_max)
{
	_id = id;

	_sl_min = new PVZoomAxisSlider(PVAbstractAxisSlider::min_value,
	                               PVAbstractAxisSlider::max_value,
	                               y_min,
	                               PVAxisSliderOrientation::Min);
	_sl_max = new PVZoomAxisSlider(PVAbstractAxisSlider::min_value,
	                               PVAbstractAxisSlider::max_value,
	                               y_max,
	                               PVAxisSliderOrientation::Max);

	_sl_min->set_owner(this);
	_sl_max->set_owner(this);

	addToGroup(_sl_min);
	addToGroup(_sl_max);

	// set positions in their parent, not in the QGraphicsScene
	_sl_min->set_value(y_min);
	_sl_max->set_value(y_max);

	connect(_sl_min, SIGNAL(slider_moved()), this, SLOT(do_sliders_moved()));
	connect(_sl_max, SIGNAL(slider_moved()), this, SLOT(do_sliders_moved()));

	PVHive::PVHive::get().register_func_observer(_sliders_manager_p,
	                                             _zsd_obs);

	PVHive::PVHive::get().register_func_observer(_sliders_manager_p,
	                                             _zsu_obs);
}

/*****************************************************************************
 * PVParallelView::PVZoomAxisSliders::remove_from_axis
 *****************************************************************************/

void PVParallelView::PVZoomAxisSliders::remove_from_axis()
{
	PVHive::call<FUNC(PVSlidersManager::del_zoom_sliders)>(_sliders_manager_p,
	                                                       _group->get_axis_id(),
	                                                       _id);
}

/*****************************************************************************
 * PVParallelView::PVZoomAxisSliders::do_sliders_moved
 *****************************************************************************/

void PVParallelView::PVZoomAxisSliders::do_sliders_moved()
{
	emit sliders_moved();

	int change = PVParallelView::PVSlidersManager::ZoomSliderNone;

	if (_sl_min->is_moving()) {
		change = PVParallelView::PVSlidersManager::ZoomSliderMin;
	}

	if (_sl_max->is_moving()) {
		change |= PVParallelView::PVSlidersManager::ZoomSliderMax;
	}

	PVHive::call<FUNC(PVSlidersManager::update_zoom_sliders)>(_sliders_manager_p,
	                                                          _group->get_axis_id(), _id,
	                                                          _sl_min->get_value(),
	                                                          _sl_max->get_value(),
	                                                          (PVParallelView::PVSlidersManager::ZoomSliderChange)change);
}

/*****************************************************************************
 * PVParallelView::PVZoomAxisSliders::zoom_sliders_del_obs::update
 *****************************************************************************/

void PVParallelView::PVZoomAxisSliders::zoom_sliders_del_obs::update(arguments_deep_copy_type const& args) const
{
	const axis_id_t &axis_id = std::get<0>(args);
	PVSlidersManager::id_t id = std::get<1>(args);

	if ((axis_id == _parent->_group->get_axis_id()) && (id == _parent->_id)) {
		_parent->group()->removeFromGroup(_parent);
		_parent->scene()->removeItem(_parent);
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomAxisSliders::zoom_sliders_update_obs::update
 *****************************************************************************/

void PVParallelView::PVZoomAxisSliders::zoom_sliders_update_obs::update(arguments_deep_copy_type const& args) const
{
	const axis_id_t &axis_id = std::get<0>(args);
	PVSlidersManager::id_t id = std::get<1>(args);

	// std::get<4>(args) (aka change) must not be used

	if ((axis_id == _parent->_group->get_axis_id()) && (id == _parent->_id)) {
		int64_t y_min = std::get<2>(args);
		int64_t y_max = std::get<3>(args);

		if (y_max < y_min) {
			std::swap(y_min, y_max);
		}

		_parent->_sl_min->set_range(PVAbstractAxisSlider::min_value, y_max);
		_parent->_sl_max->set_range(y_min, PVAbstractAxisSlider::max_value);
		_parent->refresh_value(y_min, y_max);
	}
}
