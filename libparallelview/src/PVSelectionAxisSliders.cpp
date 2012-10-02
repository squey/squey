
#include <pvparallelview/PVSelectionAxisSliders.h>
#include <pvparallelview/PVSlidersGroup.h>
#include <pvparallelview/PVSelectionAxisSlider.h>

#include <QGraphicsScene>

/*****************************************************************************
 * PVParallelView::PVSelectionAxisSliders::PVSelectionAxisSliders
 *****************************************************************************/

PVParallelView::PVSelectionAxisSliders::PVSelectionAxisSliders(QGraphicsItem *parent,
                                                               PVSlidersManager_p sm_p,
                                                               PVSlidersGroup *group) :
	PVAbstractRangeAxisSliders(parent, sm_p, group, "range selection"),
	_ssd_obs(this),
	_ssu_obs(this)
{
	PVLOG_INFO("creating selection sliders with %p\n", _sliders_manager_p.get());
}

/*****************************************************************************
 * PVParallelView::PVSelectionAxisSliders::initialize
 *****************************************************************************/

void PVParallelView::PVSelectionAxisSliders::initialize(id_t id,
                                                        int64_t y_min, int64_t y_max)
{
	_id = id;

	_sl_min = new PVSelectionAxisSlider(PVAbstractAxisSlider::min_value,
	                                    PVAbstractAxisSlider::max_value,
	                                    y_min,
	                                    PVAxisSliderOrientation::Min);
	_sl_max = new PVSelectionAxisSlider(PVAbstractAxisSlider::min_value,
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
	                                             _ssd_obs);

	PVHive::PVHive::get().register_func_observer(_sliders_manager_p,
	                                             _ssu_obs);
}

/*****************************************************************************
 * PVParallelView::PVSelectionAxisSliders::remove_from_axis
 *****************************************************************************/

void PVParallelView::PVSelectionAxisSliders::remove_from_axis()
{
	PVHive::call<FUNC(PVSlidersManager::del_selection_sliders)>(_sliders_manager_p,
	                                                            _group->get_axis_id(),
	                                                            _id);
}

/*****************************************************************************
 * PVParallelView::PVSelectionAxisSliders::do_sliders_moved
 *****************************************************************************/

void PVParallelView::PVSelectionAxisSliders::do_sliders_moved()
{
	emit sliders_moved();

	PVHive::call<FUNC(PVSlidersManager::update_selection_sliders)>(_sliders_manager_p,
	                                                               _group->get_axis_id(),
	                                                               _id,
	                                                               _sl_min->get_value(),
	                                                               _sl_max->get_value());
}

/*****************************************************************************
 * PVParallelView::PVSelectionAxisSliders::selection_sliders_del_obs::update
 *****************************************************************************/

void PVParallelView::PVSelectionAxisSliders::selection_sliders_del_obs::update(arguments_deep_copy_type const& args) const
{
	const axis_id_t &axis_id = std::get<0>(args);
	PVSlidersManager::id_t id = std::get<1>(args);

	if ((axis_id == _parent->_group->get_axis_id()) && (id == _parent->_id)) {
		_parent->group()->removeFromGroup(_parent);
		_parent->scene()->removeItem(_parent);
	}
}

/*****************************************************************************
 * PVParallelView::PVSelectionAxisSliders::selection_sliders_update_obs::update
 *****************************************************************************/

void PVParallelView::PVSelectionAxisSliders::selection_sliders_update_obs::update(arguments_deep_copy_type const& args) const
{
	const axis_id_t &axis_id = std::get<0>(args);
	PVSlidersManager::id_t id = std::get<1>(args);

	if ((axis_id == _parent->_group->get_axis_id()) && (id == _parent->_id)) {
		int64_t y_min = std::get<2>(args);
		int64_t y_max = std::get<3>(args);
		if (y_max < y_min) {
			int64_t tmp = y_max;
			y_max = y_min;
			y_min = tmp;
		}

		_parent->refresh_value(y_min, y_max);

		emit _parent->sliders_moved();
	}
}
