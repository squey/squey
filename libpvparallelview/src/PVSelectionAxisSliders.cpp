/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvparallelview/PVSelectionAxisSliders.h>
#include <pvparallelview/PVSlidersGroup.h>
#include <pvparallelview/PVSelectionAxisSlider.h>

#include <QGraphicsScene>

/*****************************************************************************
 * PVParallelView::PVSelectionAxisSliders::PVSelectionAxisSliders
 *****************************************************************************/

PVParallelView::PVSelectionAxisSliders::PVSelectionAxisSliders(QGraphicsItem* parent,
                                                               PVSlidersManager* sm_p,
                                                               PVSlidersGroup* group)
    : PVAbstractRangeAxisSliders(parent, sm_p, group, "range selection")
{
}

/*****************************************************************************
 * PVParallelView::PVSelectionAxisSliders::initialize
 *****************************************************************************/

void PVParallelView::PVSelectionAxisSliders::initialize(id_t id, int64_t y_min, int64_t y_max)
{
	_id = id;

	_sl_min =
	    new PVSelectionAxisSlider(PVAbstractAxisSlider::min_value, PVAbstractAxisSlider::max_value,
	                              y_min, PVAxisSliderOrientation::Min);
	_sl_max =
	    new PVSelectionAxisSlider(PVAbstractAxisSlider::min_value, PVAbstractAxisSlider::max_value,
	                              y_max, PVAxisSliderOrientation::Max);

	_sl_min->set_owner(this);
	_sl_max->set_owner(this);

	addToGroup(_sl_min);
	addToGroup(_sl_max);

	// set positions in their parent, not in the QGraphicsScene
	_sl_min->set_value(y_min);
	_sl_max->set_value(y_max);

	connect(_sl_min, &PVSelectionAxisSlider::slider_moved, this,
	        &PVSelectionAxisSliders::do_sliders_moved);
	connect(_sl_max, &PVSelectionAxisSlider::slider_moved, this,
	        &PVSelectionAxisSliders::do_sliders_moved);

	_sliders_manager_p->_update_selection_sliders.connect(
	    sigc::mem_fun(this, &PVParallelView::PVSelectionAxisSliders::on_selection_sliders_update));
}

/*****************************************************************************
 * PVParallelView::PVSelectionAxisSliders::remove_from_axis
 *****************************************************************************/

void PVParallelView::PVSelectionAxisSliders::remove_from_axis()
{
	_sliders_manager_p->del_selection_sliders(_group->get_col(), _id);
}

/*****************************************************************************
 * PVParallelView::PVSelectionAxisSliders::do_sliders_moved
 *****************************************************************************/

void PVParallelView::PVSelectionAxisSliders::do_sliders_moved()
{
	Q_EMIT sliders_moved();

	_sliders_manager_p->update_selection_sliders(_group->get_col(), _id, _sl_min->get_value(),
	                                             _sl_max->get_value());
}

/*****************************************************************************
 * PVParallelView::PVSelectionAxisSliders::selection_sliders_update_obs::update
 *****************************************************************************/

void PVParallelView::PVSelectionAxisSliders::on_selection_sliders_update(PVCombCol col,
                                                                         id_t id,
                                                                         int64_t y_min,
                                                                         int64_t y_max)
{
	if ((col == _group->get_col()) and (id == _id)) {
		if (y_max < y_min) {
			std::swap(y_max, y_min);
		}

		refresh_value(y_min, y_max);

		Q_EMIT sliders_moved();
	}
}
