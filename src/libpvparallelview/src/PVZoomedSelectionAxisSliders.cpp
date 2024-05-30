//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

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
	    *this, &PVParallelView::PVZoomedSelectionAxisSliders::on_zoomed_selection_sliders_update));
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
	_sliders_manager_p->del_zoomed_selection_sliders(_group->get_col(), _id);
}

/*****************************************************************************
 * PVParallelView::PVZoomedSelectionAxisSliders::do_sliders_moved
 *****************************************************************************/

void PVParallelView::PVZoomedSelectionAxisSliders::do_sliders_moved()
{
	Q_EMIT sliders_moved();

	_sliders_manager_p->update_zoomed_selection_sliders(_group->get_col(), _id,
	                                                    _sl_min->get_value(), _sl_max->get_value());
}

/*****************************************************************************
 * PVParallelView::PVZoomedSelectionAxisSliders::on_zoomed_selection_sliders_update
 *****************************************************************************/

void PVParallelView::PVZoomedSelectionAxisSliders::on_zoomed_selection_sliders_update(
    PVCombCol col, PVSlidersManager::id_t id, int64_t y_min, int64_t y_max)
{
	if ((col == _group->get_col()) && (id == _id)) {
		if (y_max < y_min) {
			std::swap(y_min, y_max);
		}

		refresh_value(y_min, y_max);

		Q_EMIT sliders_moved();
	}
}
