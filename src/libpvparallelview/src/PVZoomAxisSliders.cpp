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

#include <pvparallelview/PVZoomAxisSliders.h>
#include <pvparallelview/PVSlidersGroup.h>
#include <pvparallelview/PVZoomAxisSlider.h>

#include <QGraphicsScene>

/*****************************************************************************
 * PVParallelView::PVZoomAxisSliders::PVZoomAxisSliders
 *****************************************************************************/

PVParallelView::PVZoomAxisSliders::PVZoomAxisSliders(QGraphicsItem* parent,
                                                     PVSlidersManager* sm_p,
                                                     PVSlidersGroup* group)
    : PVAbstractRangeAxisSliders(parent, sm_p, group, "zoom")
{
}

/*****************************************************************************
 * PVParallelView::PVZoomAxisSliders::initialize
 *****************************************************************************/

void PVParallelView::PVZoomAxisSliders::initialize(id_t id, int64_t y_min, int64_t y_max)
{
	_id = id;

	_sl_min = new PVZoomAxisSlider(PVAbstractAxisSlider::min_value, PVAbstractAxisSlider::max_value,
	                               y_min, PVAxisSliderOrientation::Min);
	_sl_max = new PVZoomAxisSlider(PVAbstractAxisSlider::min_value, PVAbstractAxisSlider::max_value,
	                               y_max, PVAxisSliderOrientation::Max);

	_sl_min->set_owner(this);
	_sl_max->set_owner(this);

	addToGroup(_sl_min);
	addToGroup(_sl_max);

	// set positions in their parent, not in the QGraphicsScene
	_sl_min->set_value(y_min);
	_sl_max->set_value(y_max);

	connect(_sl_min, &PVZoomAxisSlider::slider_moved, this, &PVZoomAxisSliders::do_sliders_moved);
	connect(_sl_max, &PVZoomAxisSlider::slider_moved, this, &PVZoomAxisSliders::do_sliders_moved);

	_sliders_manager_p->_update_zoom_sliders.connect(
	    sigc::mem_fun(*this, &PVParallelView::PVZoomAxisSliders::on_zoom_sliders_update));
}

/*****************************************************************************
 * PVParallelView::PVZoomAxisSliders::remove_from_axis
 *****************************************************************************/

void PVParallelView::PVZoomAxisSliders::remove_from_axis()
{
	_sliders_manager_p->del_zoom_sliders(_group->get_col(), _id);
}

/*****************************************************************************
 * PVParallelView::PVZoomAxisSliders::do_sliders_moved
 *****************************************************************************/

void PVParallelView::PVZoomAxisSliders::do_sliders_moved()
{
	Q_EMIT sliders_moved();

	int change = PVParallelView::PVSlidersManager::ZoomSliderNone;

	if (_sl_min->is_moving()) {
		change = PVParallelView::PVSlidersManager::ZoomSliderMin;
	}

	if (_sl_max->is_moving()) {
		change |= PVParallelView::PVSlidersManager::ZoomSliderMax;
	}

	_sliders_manager_p->update_zoom_sliders(
	    _group->get_col(), _id, _sl_min->get_value(), _sl_max->get_value(),
	    (PVParallelView::PVSlidersManager::ZoomSliderChange)change);
}

/*****************************************************************************
 * PVParallelView::PVZoomAxisSliders::on_zoom_sliders_update
 *****************************************************************************/

void PVParallelView::PVZoomAxisSliders::on_zoom_sliders_update(
    PVCombCol col,
    id_t id,
    int64_t y_min,
    int64_t y_max,
    PVSlidersManager::ZoomSliderChange /*change*/)
{
	if ((col == _group->get_col()) && (id == _id)) {
		if (y_max < y_min) {
			std::swap(y_min, y_max);
		}

		_sl_min->set_range(PVAbstractAxisSlider::min_value, y_max);
		_sl_max->set_range(y_min, PVAbstractAxisSlider::max_value);
		refresh_value(y_min, y_max);
	}
}
