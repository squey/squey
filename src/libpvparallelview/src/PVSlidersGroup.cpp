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

PVParallelView::PVSlidersGroup::PVSlidersGroup(PVSlidersManager* sm_p,
                                               PVCombCol col,
                                               QGraphicsItem* parent)
    : QGraphicsItemGroup(parent), _sliders_manager_p(sm_p), _col(col), _axis_scale(1.0f)
{
	// does not care about children events
	// RH: this method is obsolete in Qt 4.8 and should be replaced with
	//     setFiltersChildEvents() but this latter does not have the same
	//     behaviour... so I keep setHandlesChildEvents()
	setHandlesChildEvents(false);

	// Populate this group with zoom slider based on the slider_manager information.
	_sliders_manager_p->iterate_zoom_sliders(
	    [&](PVCombCol col, const id_t id, const range_geometry_t& geom) {
		    if (col != _col) {
			    return;
		    }

		    add_new_zoom_sliders(id, geom.y_min, geom.y_max);
		});

	_sliders_manager_p->iterate_selection_sliders(
	    [&](PVCombCol col, const id_t id, const range_geometry_t& geom) {
		    if (col != _col) {
			    return;
		    }

		    add_new_selection_sliders(nullptr, id, geom.y_min, geom.y_max);
		});

	_sliders_manager_p->iterate_zoomed_selection_sliders(
	    [&](PVCombCol col, const id_t id, const range_geometry_t& geom) {
		    if (col != _col) {
			    return;
		    }

		    add_new_zoomed_selection_sliders(nullptr, id, geom.y_min, geom.y_max);
		});

	// Connect the slider_manager to update this group with slider_manager update.
	_sliders_manager_p->_new_zoom_sliders.connect(
	    sigc::mem_fun(*this, &PVParallelView::PVSlidersGroup::on_new_zoom_slider));
	_sliders_manager_p->_new_selection_sliders.connect(
	    sigc::mem_fun(*this, &PVParallelView::PVSlidersGroup::on_new_selection_sliders));
	_sliders_manager_p->_new_zoomed_selection_sliders.connect(
	    sigc::mem_fun(*this, &PVParallelView::PVSlidersGroup::on_new_zoomed_selection_sliders));

	_sliders_manager_p->_del_zoom_sliders.connect(
	    sigc::mem_fun(*this, &PVParallelView::PVSlidersGroup::on_del_zoom_sliders));
	_sliders_manager_p->_del_selection_sliders.connect(
	    sigc::mem_fun(*this, &PVParallelView::PVSlidersGroup::on_del_selection_sliders));
	_sliders_manager_p->_del_zoomed_selection_sliders.connect(
	    sigc::mem_fun(*this, &PVParallelView::PVSlidersGroup::on_del_zoomed_selection_sliders));
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

	for (const auto& it : sasliders) {
		del_selection_sliders(it.first);
	}

	zsas_set_t zssliders = _zoomed_selection_sliders;

	for (const auto& it : zssliders) {
		del_zoomed_selection_sliders(it.first);
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::remove_zoom_slider
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::remove_zoom_slider()
{
	zas_set_t zasliders = _zoom_sliders;

	for (const auto& it : zasliders) {
		if (it.first == (id_t) this) {
			del_zoom_sliders(it.first);
		}
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::delete_own_zoom_slider
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::delete_own_zoom_slider()
{
	_sliders_manager_p->del_zoom_sliders(_col, this);
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::set_axis_scale
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::set_axis_scale(float s)
{
	_axis_scale = s;

	for (auto& it : _selection_sliders) {
		it.second->refresh();
	}

	for (auto& it : _zoomed_selection_sliders) {
		it.second->refresh();
	}

	for (auto& it : _zoom_sliders) {
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

	for (const auto& it : _selection_sliders) {
		ranges.push_back(it.second->get_range());
	}

	for (const auto& it : _zoomed_selection_sliders) {
		ranges.push_back(it.second->get_range());
	}

	return ranges;
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::grow_selection_sliders_to_axis
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::rescale_selection_sliders(range_t before, range_t after)
{
	const double before_span = double(before.second) - double(before.first);
	const double after_span = double(after.second) - double(after.first);

	// Carried from the middle of the band it framed onto the middle of the band
	// those rows occupy now, stretched by how much that band grew. Either band can
	// be a single value rather than a stretch -- a selection of one row lands
	// wherever its axis puts a lone value -- and the pair then keeps its height
	// and simply moves onto the rows. See
	// PVFullParallelViewSelectionRectangle::rescale_vertically.
	const double factor = (before_span > 0. and after_span > 0.) ? after_span / before_span : 1.;

	const double before_middle = (double(before.first) + double(before.second)) / 2.;
	const double after_middle = (double(after.first) + double(after.second)) / 2.;

	const auto carry = [&](int64_t value) {
		const double moved = after_middle + (double(value) - before_middle) * factor;
		return PVCore::clamp<int64_t>((int64_t)moved, PVAbstractAxisSlider::min_value,
		                              PVAbstractAxisSlider::max_value);
	};

	const auto rescale = [&](auto& sliders_set) {
		for (const auto& it : sliders_set) {
			const range_t range = it.second->get_range();
			const int64_t y_min = carry(range.first);
			const int64_t y_max = carry(range.second);

			if (y_min == range.first and y_max == range.second) {
				continue;
			}

			it.second->set_range_quietly(y_min, y_max);
			_sliders_manager_p->set_selection_sliders_geometry(_col, it.first, y_min, y_max);
		}
	};

	rescale(_selection_sliders);
	rescale(_zoomed_selection_sliders);
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::add_zoom_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::add_zoom_sliders(int64_t y_min, int64_t y_max)
{
	_sliders_manager_p->new_zoom_sliders(_col, this, y_min * BUCKET_ELT_COUNT,
	                                     y_max * BUCKET_ELT_COUNT);
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::add_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::add_selection_sliders(int64_t y_min, int64_t y_max)
{
	y_min *= BUCKET_ELT_COUNT;
	y_max *= BUCKET_ELT_COUNT;

	auto sliders = new PVParallelView::PVSelectionAxisSliders(this, _sliders_manager_p, this);
	add_new_selection_sliders(sliders, sliders, y_min, y_max);

	_sliders_manager_p->new_selection_sliders(_col, sliders, y_min, y_max);
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::add_zoomed_selection_sliders
 *****************************************************************************/

PVParallelView::PVZoomedSelectionAxisSliders*
PVParallelView::PVSlidersGroup::add_zoomed_selection_sliders(int64_t y_min, int64_t y_max)
{
	auto sliders = new PVZoomedSelectionAxisSliders(this, _sliders_manager_p, this);
	add_new_zoomed_selection_sliders(sliders, sliders, y_min, y_max);

	_sliders_manager_p->new_zoomed_selection_sliders(_col, sliders, y_min, y_max);

	return sliders;
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::del_zoom_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::del_zoom_sliders(id_t id)
{
	auto it = _zoom_sliders.find(id);

	if (it != _zoom_sliders.end()) {
		removeFromGroup(it->second);
		if (scene()) {
			scene()->removeItem(it->second);
		}
		delete it->second;
		_zoom_sliders.erase(it);
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::del_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::del_selection_sliders(id_t id)
{

	auto it = _selection_sliders.find(id);

	if (it != _selection_sliders.end()) {
		removeFromGroup(it->second);
		if (scene()) {
			scene()->removeItem(it->second);
		}
		delete it->second;
		_selection_sliders.erase(it);
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::del_zoomed_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::del_zoomed_selection_sliders(id_t id)
{
	auto it = _zoomed_selection_sliders.find(id);

	if (it != _zoomed_selection_sliders.end()) {
		removeFromGroup(it->second);
		if (scene()) {
			scene()->removeItem(it->second);
		}
		delete it->second;
		_zoomed_selection_sliders.erase(it);
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::sliders_moving
 *****************************************************************************/

bool PVParallelView::PVSlidersGroup::sliders_moving() const
{
	for (const auto& it : _selection_sliders) {
		if (it.second->is_moving()) {
			return true;
		}
	}
	for (const auto& it : _zoomed_selection_sliders) {
		if (it.second->is_moving()) {
			return true;
		}
	}
	for (const auto& it : _zoom_sliders) {
		if (it.second->is_moving()) {
			return true;
		}
	}
	return false;
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::new_zoom_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::add_new_zoom_sliders(id_t id, int64_t y_min, int64_t y_max)
{
	auto sliders = new PVParallelView::PVZoomAxisSliders(this, _sliders_manager_p, this);

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

void PVParallelView::PVSlidersGroup::add_new_selection_sliders(
    PVParallelView::PVSelectionAxisSliders* sliders, id_t id, int64_t y_min, int64_t y_max)
{
	if (sliders == nullptr) {
		sliders = new PVParallelView::PVSelectionAxisSliders(this, _sliders_manager_p, this);
	}

	if (id == nullptr) {
		id = sliders;
	}

	sliders->initialize(id, y_min, y_max);

	addToGroup(sliders);

	sliders->setPos(0, 0);

	connect(sliders, &PVSelectionAxisSliders::sliders_moved, this,
	        &PVSlidersGroup::selection_slider_moved);

	_selection_sliders[id] = sliders;
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::add_new_zoomed_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::add_new_zoomed_selection_sliders(
    PVParallelView::PVZoomedSelectionAxisSliders* sliders, id_t id, int64_t y_min, int64_t y_max)
{
	if (sliders == nullptr) {
		sliders = new PVZoomedSelectionAxisSliders(this, _sliders_manager_p, this);
	}

	if (id == nullptr) {
		id = sliders;
	}

	sliders->initialize(id, y_min, y_max);

	addToGroup(sliders);

	sliders->setPos(0, 0);

	connect(sliders, &PVZoomedSelectionAxisSliders::sliders_moved, this,
	        &PVSlidersGroup::selection_slider_moved);

	_zoomed_selection_sliders[id] = sliders;
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::on_new_zoom_slider
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::on_new_zoom_slider(PVCombCol col,
                                                        PVSlidersManager::id_t id,
                                                        int64_t y_min,
                                                        int64_t y_max)
{
	if (col == _col) {
		if (id != this) {
			add_new_zoom_sliders(id, y_min, y_max);
		}
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::on_new_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::on_new_selection_sliders(PVCombCol col,
                                                              PVSlidersManager::id_t id,
                                                              int64_t y_min,
                                                              int64_t y_max)
{
	if (col == _col) {
		if (_selection_sliders.find(id) == _selection_sliders.end()) {
			add_new_selection_sliders(nullptr, id, y_min, y_max);
		}
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::on_new_zoomed_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::on_new_zoomed_selection_sliders(PVCombCol col,
                                                                     PVSlidersManager::id_t id,
                                                                     int64_t y_min,
                                                                     int64_t y_max)
{
	if (col == _col) {
		if (_zoomed_selection_sliders.find(id) == _zoomed_selection_sliders.end()) {
			add_new_zoomed_selection_sliders(nullptr, id, y_min, y_max);
		}
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::on_del_zoom_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::on_del_zoom_sliders(PVCombCol col, PVSlidersManager::id_t id)
{
	if (col == _col) {
		del_zoom_sliders(id);
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::on_del_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::on_del_selection_sliders(PVCombCol col,
                                                              PVSlidersManager::id_t id)
{
	if (col == _col) {
		del_selection_sliders(id);
	}
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::on_del_zoomed_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::on_del_zoomed_selection_sliders(PVCombCol col,
                                                                     PVSlidersManager::id_t id)
{
	if (col == _col) {
		del_zoomed_selection_sliders(id);
	}
}

QRectF PVParallelView::PVSlidersGroup::boundingRect() const
{
	return childrenBoundingRect();
}
