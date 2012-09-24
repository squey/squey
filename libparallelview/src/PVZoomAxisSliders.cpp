
#include <pvparallelview/PVZoomAxisSliders.h>
#include <pvparallelview/PVSlidersGroup.h>

#include <QGraphicsScene>

/*****************************************************************************
 * PVParallelView::PVZoomAxisSliders::PVZoomAxisSliders
 *****************************************************************************/

PVParallelView::PVZoomAxisSliders::PVZoomAxisSliders(QGraphicsItem *parent,
                                                     PVSlidersGroup *group) :
	PVAbstractAxisSliders(parent, group),
	_zsd_obs(this),
	_zsu_obs(this)
{
	setHandlesChildEvents(false);
}


/*****************************************************************************
 * PVParallelView::PVZoomAxisSliders::initialize
 *****************************************************************************/

void PVParallelView::PVZoomAxisSliders::initialize(PVParallelView::PVSlidersManager_p sm_p,
                                                   id_t id,
                                                   uint32_t y_min, uint32_t y_max)
{
	_sliders_manager_p = sm_p;
	_id = id;

	_sl_min = new PVAxisSlider(0, 1024, y_min);
	_sl_max = new PVAxisSlider(0, 1024, y_max);

	addToGroup(_sl_min);
	addToGroup(_sl_max);

	// set positions in their parent, not in the QGraphicsScene
	_sl_min->setPos(0, y_min);
	_sl_max->setPos(0, y_max);

	connect(_sl_min, SIGNAL(slider_moved()), this, SLOT(do_sliders_moved()));
	connect(_sl_max, SIGNAL(slider_moved()), this, SLOT(do_sliders_moved()));

	PVHive::PVHive::get().register_func_observer(sm_p,
	                                             _zsd_obs);

	PVHive::PVHive::get().register_func_observer(sm_p,
	                                             _zsu_obs);
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
	                                                          _group->get_axis_index(), _id,
	                                                          _sl_min->value(),
	                                                          _sl_max->value(),
	                                                          (PVParallelView::PVSlidersManager::ZoomSliderChange)change);
}

/*****************************************************************************
 * PVParallelView::PVZoomAxisSliders::zoom_sliders_del_obs::update
 *****************************************************************************/

void PVParallelView::PVZoomAxisSliders::zoom_sliders_del_obs::update(arguments_deep_copy_type const& args) const
{
	PVZoneID axis_index = std::get<0>(args);
	PVSlidersManager::id_t id = std::get<1>(args);

	if ((axis_index == _parent->_group->get_axis_index()) && (id == _parent->_id)) {
		_parent->group()->removeFromGroup(_parent);
		_parent->scene()->removeItem(_parent);
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomAxisSliders::zoom_sliders_update_obs::update
 *****************************************************************************/

void PVParallelView::PVZoomAxisSliders::zoom_sliders_update_obs::update(arguments_deep_copy_type const& args) const
{
	PVZoneID axis_index = std::get<0>(args);
	PVSlidersManager::id_t id = std::get<1>(args);

	// std::get<4>(args) (aka change) must not be used

	if ((axis_index == _parent->_group->get_axis_index()) && (id == _parent->_id)) {
		int y_min = std::get<2>(args);
		int y_max = std::get<3>(args);
		if (y_max < y_min) {
			int tmp = y_max;
			y_max = y_min;
			y_min = tmp;
		}

		_parent->_sl_min->set_range(0, y_max);
		_parent->_sl_min->set_value(y_min);
		_parent->_sl_max->set_range(y_min, 1024);
		_parent->_sl_max->set_value(y_max);
	}
}
