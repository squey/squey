
#include <pvparallelview/PVZoomedSelectionAxisSliders.h>
#include <pvparallelview/PVSlidersGroup.h>
#include <pvparallelview/PVZoomedSelectionAxisSlider.h>

#include <QGraphicsScene>

/*****************************************************************************
 * PVParallelView::PVZoomedSelectionAxisSliders::PVZoomedSelectionAxisSliders
 *****************************************************************************/

PVParallelView::PVZoomedSelectionAxisSliders::PVZoomedSelectionAxisSliders(QGraphicsItem *parent,
	     PVSlidersManager_p sm_p,
	     PVSlidersGroup *group) :
	PVAbstractRangeAxisSliders(parent, sm_p, group, "range selection"),
	_zssu_obs(this)
{
}

/*****************************************************************************
 * PVParallelView::PVZoomedSelectionAxisSliders::initialize
 *****************************************************************************/

void PVParallelView::PVZoomedSelectionAxisSliders::initialize(id_t id,
                                                              int64_t y_min, int64_t y_max)
{
	_id = id;

	_sl_min = new PVZoomedSelectionAxisSlider(PVAbstractAxisSlider::min_value,
	                                          PVAbstractAxisSlider::max_value,
	                                          y_min,
	                                          PVAxisSliderOrientation::Min);
	_sl_max = new PVZoomedSelectionAxisSlider(PVAbstractAxisSlider::min_value,
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
	                                             _zssu_obs);
}

/*****************************************************************************
 * PVParallelView::PVZoomedSelectionAxisSliders::set_value
 *****************************************************************************/

void PVParallelView::PVZoomedSelectionAxisSliders::set_value(int64_t y_min, int64_t y_max)
{
	refresh_value(y_min, y_max);
	do_sliders_moved();
}

/*****************************************************************************
 * PVParallelView::PVZoomedSelectionAxisSliders::remove_from_axis
 *****************************************************************************/

void PVParallelView::PVZoomedSelectionAxisSliders::remove_from_axis()
{
	PVHive::call<FUNC(PVSlidersManager::del_zoomed_selection_sliders)>(_sliders_manager_p,
	                                                                   _group->get_axis_id(),
	                                                                   _id);
}

/*****************************************************************************
 * PVParallelView::PVZoomedSelectionAxisSliders::do_sliders_moved
 *****************************************************************************/

void PVParallelView::PVZoomedSelectionAxisSliders::do_sliders_moved()
{
	emit sliders_moved();

	PVHive::call<FUNC(PVSlidersManager::update_zoomed_selection_sliders)>(_sliders_manager_p,
	                                                                      _group->get_axis_id(),
	                                                                      _id,
	                                                                      _sl_min->get_value(),
	                                                                      _sl_max->get_value());
}

/*****************************************************************************
 * PVParallelView::PVZoomedSelectionAxisSliders::zoomed_selection_sliders_update_obs::update
 *****************************************************************************/

void PVParallelView::PVZoomedSelectionAxisSliders::zoomed_selection_sliders_update_obs::update(arguments_deep_copy_type const& args) const
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
