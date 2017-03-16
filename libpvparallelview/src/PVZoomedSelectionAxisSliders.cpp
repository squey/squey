/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvparallelview/PVZoomedSelectionAxisSliders.h>
#include <pvparallelview/PVSlidersGroup.h>
#include <pvparallelview/PVZoomedSelectionAxisSlider.h>

#include <QGraphicsScene>

/*****************************************************************************
 * PVParallelView::PVZoomedSelectionAxisSliders::PVZoomedSelectionAxisSliders
 *****************************************************************************/

PVParallelView::PVZoomedSelectionAxisSliders::PVZoomedSelectionAxisSliders(QGraphicsItem* parent,
                                                                           PVSlidersManager* sm_p,
                                                                           PVSlidersGroup* group)
    : PVAbstractRangeAxisSliders(parent, sm_p, group, "range selection")
{
}

/*****************************************************************************
 * PVParallelView::PVZoomedSelectionAxisSliders::initialize
 *****************************************************************************/

void PVParallelView::PVZoomedSelectionAxisSliders::initialize(id_t id, int64_t y_min, int64_t y_max)
{
	_id = id;

	_sl_min = new PVZoomedSelectionAxisSlider(PVAbstractAxisSlider::min_value,
	                                          PVAbstractAxisSlider::max_value, y_min,
	                                          PVAxisSliderOrientation::Min);
	_sl_max = new PVZoomedSelectionAxisSlider(PVAbstractAxisSlider::min_value,
	                                          PVAbstractAxisSlider::max_value, y_max,
	                                          PVAxisSliderOrientation::Max);

	_sl_min->set_owner(this);
	_sl_max->set_owner(this);

	addToGroup(_sl_min);
	addToGroup(_sl_max);

	// set positions in their parent, not in the QGraphicsScene
	_sl_min->set_value(y_min);
	_sl_max->set_value(y_max);

	connect(_sl_min, &PVZoomedSelectionAxisSlider::slider_moved, this,
	        &PVZoomedSelectionAxisSliders::do_sliders_moved);
	connect(_sl_max, &PVZoomedSelectionAxisSlider::slider_moved, this,
	        &PVZoomedSelectionAxisSliders::do_sliders_moved);

	_sliders_manager_p->_update_zoomed_selection_sliders.connect(sigc::mem_fun(
	    this, &PVParallelView::PVZoomedSelectionAxisSliders::on_zoomed_selection_sliders_update));
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
	_sliders_manager_p->del_zoomed_selection_sliders(_group->get_nraw_col(), _id);
}

/*****************************************************************************
 * PVParallelView::PVZoomedSelectionAxisSliders::do_sliders_moved
 *****************************************************************************/

void PVParallelView::PVZoomedSelectionAxisSliders::do_sliders_moved()
{
	Q_EMIT sliders_moved();

	_sliders_manager_p->update_zoomed_selection_sliders(_group->get_nraw_col(), _id,
	                                                    _sl_min->get_value(), _sl_max->get_value());
}

/*****************************************************************************
 * PVParallelView::PVZoomedSelectionAxisSliders::on_zoomed_selection_sliders_update
 *****************************************************************************/

void PVParallelView::PVZoomedSelectionAxisSliders::on_zoomed_selection_sliders_update(
    PVCol nraw_col, PVSlidersManager::id_t id, int64_t y_min, int64_t y_max)
{
	if ((nraw_col == _group->get_nraw_col()) && (id == _id)) {
		if (y_max < y_min) {
			std::swap(y_min, y_max);
		}

		refresh_value(y_min, y_max);

		Q_EMIT sliders_moved();
	}
}
