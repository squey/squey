
#include <pvparallelview/PVSelectionAxisSliders.h>

#include <QGraphicsScene>

/*****************************************************************************
 * PVParallelView::PVSelectionAxisSliders::PVSelectionAxisSliders
 *****************************************************************************/

PVParallelView::PVSelectionAxisSliders::PVSelectionAxisSliders(QGraphicsItem *parent) :
	PVAbstractAxisSliders(parent),
	_ssd_obs(this),
	_ssu_obs(this)
{
	setHandlesChildEvents(false);
}


/*****************************************************************************
 * PVParallelView::PVSelectionAxisSliders::initialize
 *****************************************************************************/

void PVParallelView::PVSelectionAxisSliders::initialize(PVParallelView::PVSlidersManager_p sm_p,
                                                        PVCol axis, id_t id,
                                                        uint32_t y_min, uint32_t y_max)
{
	_sliders_manager_p = sm_p;
	_axis = axis;
	_id = id;

	_sl_min = new PVAxisSlider(0, 1024, y_min);
	_sl_max = new PVAxisSlider(0, 1024, y_max);

	addToGroup(_sl_min);
	addToGroup(_sl_max);

	_sl_min->setPos(0, 0);
	_sl_max->setPos(0, 0);

	connect(_sl_min, SIGNAL(slider_moved()), this, SLOT(do_sliders_moved()));
	connect(_sl_max, SIGNAL(slider_moved()), this, SLOT(do_sliders_moved()));

	PVHive::PVHive::get().register_func_observer(sm_p,
	                                             _ssd_obs);

	PVHive::PVHive::get().register_func_observer(sm_p,
	                                             _ssu_obs);
}

/*****************************************************************************
 * PVParallelView::PVSelectionAxisSliders::selection_sliders_del_obs::update
 *****************************************************************************/

void PVParallelView::PVSelectionAxisSliders::selection_sliders_del_obs::update(arguments_deep_copy_type const& args) const
{
	PVCol axis = std::get<0>(args);
	PVSlidersManager::id_t id = std::get<1>(args);

	if ((axis == _parent->_axis) && (id == _parent->_id)) {
		_parent->group()->removeFromGroup(_parent);
		_parent->scene()->removeItem(_parent);
	}
}

/*****************************************************************************
 * PVParallelView::PVSelectionAxisSliders::selection_sliders_update_obs::update
 *****************************************************************************/

void PVParallelView::PVSelectionAxisSliders::selection_sliders_update_obs::update(arguments_deep_copy_type const& args) const
{
	PVCol axis = std::get<0>(args);
	PVSlidersManager::id_t id = std::get<1>(args);

	if ((axis == _parent->_axis) && (id == _parent->_id)) {
		uint32_t y_min = std::get<2>(args);
		uint32_t y_max = std::get<3>(args);

		_parent->_sl_min->set_value(y_min);
		_parent->_sl_max->set_value(y_max);
		emit _parent->sliders_moved();
	}
}
