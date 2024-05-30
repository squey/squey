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

#include <cmath>

#include <pvparallelview/PVBCIDrawingBackend.h>
#include <pvparallelview/PVLinesView.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVRenderingPipeline.h>
#include <pvparallelview/PVZoneRenderingBCI.h>
#include <pvparallelview/PVZoneTree.h>

#include <QObject>
#include <QMetaMethod>
#include <QThread>
#include <QDebug>

constexpr static int zoom_divisor = 5;
constexpr static double zoom_root_value =
    1.148698354997035; // pow(2.0, 1.0 / zoom_divisor); with divisiro = 5;
constexpr static int32_t min_zoom_level =
    (boost::static_log2<PVParallelView::ZoneMinWidth>::value -
     boost::static_log2<PVParallelView::ZoneBaseWidth>::value) *
    zoom_divisor;
constexpr static int32_t max_zoom_level =
    (boost::static_log2<PVParallelView::ZoneMaxWidth>::value -
     boost::static_log2<PVParallelView::ZoneBaseWidth>::value) *
    zoom_divisor;

static int zoom_level_to_width(int32_t zoom_level, int base_width = PVParallelView::ZoneBaseWidth)
{
	zoom_level = PVCore::clamp<int32_t>(zoom_level, min_zoom_level, max_zoom_level);

	int32_t primary_zoom_level = zoom_level / zoom_divisor;
	int32_t secondary_zoom_level = zoom_level % zoom_divisor;

	uint32_t width =
	    base_width * pow(2.0, primary_zoom_level) * pow(zoom_root_value, secondary_zoom_level);

	return PVCore::clamp<uint32_t>(width, PVParallelView::ZoneMinWidth,
	                               PVParallelView::ZoneMaxWidth);
}

PVParallelView::PVLinesView::PVLinesView(PVBCIDrawingBackend& backend,
                                         PVZonesManager const& zm,
                                         PVZonesProcessor& zp_sel,
                                         PVZonesProcessor& zp_bg,
                                         QObject* img_update_receiver,
                                         uint32_t zone_width)
    : _backend(backend)
    , _first_zone(0)
    , _last_zone(0)
    , _img_update_receiver(img_update_receiver)
    , _processor_sel(zp_sel)
    , _processor_bg(zp_bg)
    , _visible_view_x(0)
    , _visible_view_width(0)
    , _zm(zm)
    , _zone_max_width(zone_width)
{
	const auto nb_of_managed_zones = get_number_of_managed_zones();
	// We initialize all zones ZoneWidthWithZoomLevel
	_list_of_zone_width_with_zoom_level.resize(nb_of_managed_zones);
}

void PVParallelView::PVLinesView::call_refresh_slots(size_t zone_index)
{
	// Call both zr_sel_finished and zr_bg_finished slots on _img_update_receiver
	if (!_img_update_receiver) {
		return;
	}

	QMetaObject::invokeMethod(_img_update_receiver, "zr_sel_finished", Qt::QueuedConnection,
	                          Q_ARG(PVParallelView::PVZoneRendering_p, PVZoneRendering_p()),
	                          Q_ARG(size_t, zone_index));
	QMetaObject::invokeMethod(_img_update_receiver, "zr_bg_finished", Qt::QueuedConnection,
	                          Q_ARG(PVParallelView::PVZoneRendering_p, PVZoneRendering_p()),
	                          Q_ARG(size_t, zone_index));
}

void PVParallelView::PVLinesView::cancel_and_wait_all_rendering()
{
	for (SingleZoneImages& single_zone_images : _list_of_single_zone_images) {
		single_zone_images.cancel_all_and_wait();
	}
}

void PVParallelView::PVLinesView::connect_zr(PVZoneRenderingBCIBase* zr, const char* slot)
{
	if (_img_update_receiver) {
		zr->set_render_finished_slot(_img_update_receiver, slot);
	}
}

uint32_t
PVParallelView::PVLinesView::get_left_border_position_of_zone_in_scene(size_t zone_index) const
{
	assert(zone_index < _list_of_zone_width_with_zoom_level.size());

	// The first zone start in scene after the first axis which is at absciss 0
	uint32_t pos = get_axis_width();

	// We do stop after the right axis of the previous zone
	for (size_t zin = 0; zin < zone_index; ++zin) {
		pos += get_zone_width(zin) + get_axis_width();
	}

	return pos;
}

uint32_t PVParallelView::PVLinesView::get_right_border_of_scene() const
{
	return get_left_border_position_of_zone_in_scene(_list_of_zone_width_with_zoom_level.size() -
	                                                 1) +
	       get_zone_width(_list_of_zone_width_with_zoom_level.size() - 1) + get_axis_width();
}

size_t PVParallelView::PVLinesView::get_number_of_managed_zones() const
{
	return get_zones_manager().get_number_of_axes_comb_zones();
}

size_t PVParallelView::PVLinesView::get_zone_index_from_scene_pos(int32_t abs_pos) const
{
	size_t zone_index(0);
	ssize_t pos = 0;
	for (; zone_index < _list_of_zone_width_with_zoom_level.size() - 1; ++zone_index) {
		pos += get_axis_width() + get_zone_width(zone_index);
		if (pos > abs_pos) {
			break;
		}
	}

	return std::min(zone_index, _list_of_zone_width_with_zoom_level.size() - 1);
}

uint32_t PVParallelView::PVLinesView::get_zone_width(size_t zone_index) const
{
	assert(zone_index < _list_of_zone_width_with_zoom_level.size());
	uint32_t width = _list_of_zone_width_with_zoom_level[zone_index].get_width();
	return width;
}

void PVParallelView::PVLinesView::increase_base_zoom_level_of_zone(size_t zone_index)
{
	assert(zone_index < _list_of_zone_width_with_zoom_level.size());
	_list_of_zone_width_with_zoom_level[zone_index].increase_zoom_level();
}

void PVParallelView::PVLinesView::increase_global_zoom_level()
{
	for (ZoneWidthWithZoomLevel& z : _list_of_zone_width_with_zoom_level) {
		z.increase_zoom_level();
	}
}

void PVParallelView::PVLinesView::decrease_base_zoom_level_of_zone(size_t zone_index)
{
	assert(zone_index < _list_of_zone_width_with_zoom_level.size());
	_list_of_zone_width_with_zoom_level[zone_index].decrease_zoom_level();
}

void PVParallelView::PVLinesView::decrease_global_zoom_level()
{
	for (ZoneWidthWithZoomLevel& z : _list_of_zone_width_with_zoom_level) {
		z.decrease_zoom_level();
	}
}

bool PVParallelView::PVLinesView::initialize_zones_width(int view_width)
{
	bool fit_in_view = false;

	_visible_view_width = view_width;

	// reduding the width to add a margin
	view_width *= 0.95;

	int zones_number = int(get_number_of_managed_zones());

	int zone_width = PVParallelView::ZoneDefaultWidth;

	if ((zones_number * zone_width) < view_width) {
		// there is still empty space, growing zones width to fit in
		zone_width = view_width / zones_number;
		fit_in_view = true;
	}

	reset_zones_width(zone_width);

	return fit_in_view;
}

void PVParallelView::PVLinesView::render_all_zones_images(int32_t view_x,
                                                          uint32_t view_width,
                                                          const float zoom_y)
{
	set_new_view(view_x, view_width);
	visit_all_zones_to_render([&](size_t zone_index) {
		assert(is_zone_drawn(zone_index));
		render_single_zone_images(zone_index, zoom_y);
	});
}

void PVParallelView::PVLinesView::render_all_zones_bg_image(int32_t view_x,
                                                            uint32_t view_width,
                                                            const float zoom_y)
{
	set_new_view(view_x, view_width);
	visit_all_zones_to_render([&](size_t zone_index) {
		assert(is_zone_drawn(zone_index));
		render_single_zone_bg_image(zone_index, zoom_y);
	});
}

void PVParallelView::PVLinesView::render_all_zones_sel_image(int32_t view_x,
                                                             uint32_t view_width,
                                                             const float zoom_y)
{
	set_new_view(view_x, view_width);
	visit_all_zones_to_render([&](size_t zone_index) {
		assert(is_zone_drawn(zone_index));
		render_single_zone_sel_image(zone_index, zoom_y);
	});
}

void PVParallelView::PVLinesView::render_single_zone_images(size_t zone_index, const float zoom_y)
{
	assert(is_zone_drawn(zone_index));
	render_single_zone_bg_image(zone_index, zoom_y);
	render_single_zone_sel_image(zone_index, zoom_y);
}

void PVParallelView::PVLinesView::render_single_zone_bg_image(size_t zone_index, const float zoom_y)
{
	assert(is_zone_drawn(zone_index));
	assert(not _img_update_receiver or QThread::currentThread() == _img_update_receiver->thread());

	const size_t zone_offset = get_zone_index_offset(zone_index);
	SingleZoneImages& single_zone_images = get_single_zone_images(zone_offset);
	// single_zone_images.cancel_last_bg();
	const uint32_t width = get_zone_width(zone_index);
	single_zone_images.bg->set_width(width);

	PVZoneRenderingBCI_p<PARALLELVIEW_ZT_BBITS> zr(new PVZoneRenderingBCI<PARALLELVIEW_ZT_BBITS>(
	    _zm.get_zone_id(zone_index),
	    [&](PVZoneID zone_id, PVCore::PVHSVColor const* colors,
	        PVBCICode<PARALLELVIEW_ZT_BBITS>* codes) {
		    return this->get_zones_manager().get_zone_tree(zone_id).browse_tree_bci(colors, codes);
	    },
	    single_zone_images.bg, 0, width, zoom_y,
	    false // not reversed
	    ));

	connect_zr(zr.get(), "zr_bg_finished");

	PVZoneRenderingBCIBase_p last_zr = single_zone_images.last_zr_bg;
	if (last_zr) {
		last_zr->cancel_and_add_job(_processor_bg, zr);
	} else {
		_processor_bg.add_job(zr);
	}
	single_zone_images.last_zr_bg = zr;
}

void PVParallelView::PVLinesView::render_single_zone_sel_image(size_t zone_index,
                                                               const float zoom_y)
{
	assert(is_zone_drawn(zone_index));
	assert(not _img_update_receiver or QThread::currentThread() == _img_update_receiver->thread());

	const size_t zone_offset = get_zone_index_offset(zone_index);
	SingleZoneImages& single_zone_images = get_single_zone_images(zone_offset);
	// single_zone_images.cancel_last_sel();
	const uint32_t width = get_zone_width(zone_index);
	single_zone_images.sel->set_width(width);

	PVZoneRenderingBCI_p<PARALLELVIEW_ZT_BBITS> zr(new PVZoneRenderingBCI<PARALLELVIEW_ZT_BBITS>(
	    _zm.get_zone_id(zone_index),
	    [&](PVZoneID zone_id, PVCore::PVHSVColor const* colors,
	        PVBCICode<PARALLELVIEW_ZT_BBITS>* codes) {
		    return this->get_zones_manager().get_zone_tree(zone_id).browse_tree_bci_sel(colors,
		                                                                                codes);
	    },
	    single_zone_images.sel, 0, width, zoom_y,
	    false // not reversed
	    ));

	connect_zr(zr.get(), "zr_sel_finished");

	PVZoneRenderingBCIBase_p last_zr = single_zone_images.last_zr_sel;
	if (last_zr) {
		last_zr->cancel_and_add_job(_processor_sel, zr);
	} else {
		_processor_sel.add_job(zr);
	}
	single_zone_images.last_zr_sel = zr;
}

size_t PVParallelView::PVLinesView::set_new_view(int32_t new_view_x, uint32_t view_width)
{
	// Change view_x
	_visible_view_x = new_view_x;
	_visible_view_width = view_width;

	// and set new first zone
	size_t previous_first_zone = _first_zone;
	_first_zone = get_zone_index_from_scene_pos(new_view_x);
	_last_zone = get_zone_index_from_scene_pos(new_view_x + view_width);

	set_nb_drawable_zones(get_number_of_visible_zones());

	qDebug() << "set_new_view: " << get_number_of_visible_zones() << "("
	         << get_first_visible_zone_index() << "->" << get_last_visible_zone_index() << ")";

	// Returns the previous first zone index
	return previous_first_zone;
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::set_nb_drawable_zones
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::set_nb_drawable_zones(size_t nb_zones)
{
	if (nb_zones > _list_of_single_zone_images.size()) {
		for (size_t z = _list_of_single_zone_images.size(); z < nb_zones; ++z) {
			_list_of_single_zone_images.emplace_back(this->backend(), _zone_max_width);
		}
	} else if (nb_zones < _list_of_single_zone_images.size()) {
		_list_of_single_zone_images.resize(nb_zones);
	}
	assert(_list_of_single_zone_images.size() == nb_zones);
}

void PVParallelView::PVLinesView::set_axis_width(uint32_t width)
{
	_axis_width = width;
}

void PVParallelView::PVLinesView::translate(int32_t view_x, uint32_t view_width, const float zoom_y)
{
	const size_t previous_first_zone = _first_zone;
	const size_t previous_last_zone = _last_zone;
	auto zone_images_copy = _list_of_single_zone_images;

	// First, set new view x (before launching anything in the future !! ;))
	set_new_view(view_x, view_width);

	if (previous_first_zone == _first_zone && previous_last_zone == _last_zone) {
		// "Le changement, c'est pas maintenant !"
		return;
	}

	auto fzone_draw = [this, zoom_y](size_t zone_index) {
		assert(is_zone_drawn(zone_index));
		render_single_zone_images(zone_index, zoom_y);
	};

	if (_first_zone > previous_last_zone or _last_zone < previous_first_zone) {
		visit_all_zones_to_render(fzone_draw);
		return;
	}

	auto keep_zones = [this](size_t keep_begin, size_t keep_end) {
		if (_img_update_receiver) {
			for (size_t zone_index = keep_begin; zone_index < keep_end; ++zone_index) {
				call_refresh_slots(zone_index);
			}
		}
	};
	auto draw_zones = [fzone_draw](size_t draw_begin, size_t draw_end) {
		for (size_t zone_index = draw_begin; zone_index < draw_end; ++zone_index) {
			fzone_draw(zone_index);
		}
	};

	// We test whether translation happened on the left or on the right.
	if (_first_zone > previous_first_zone or _last_zone > previous_last_zone) {
		// The scene was translated to the left
		if (previous_first_zone != _first_zone) {
			std::rotate(zone_images_copy.begin(),
			            zone_images_copy.begin() + _first_zone - previous_first_zone,
			            zone_images_copy.end());
			std::swap_ranges(
			    zone_images_copy.begin(),
			    zone_images_copy.begin() +
			        std::min<size_t>(_list_of_single_zone_images.size(), zone_images_copy.size()),
			    _list_of_single_zone_images.begin());
		}
		keep_zones(_first_zone, previous_last_zone + 1);
		draw_zones(previous_last_zone + 1, _last_zone + 1);
	} else {
		// The scene was translated to the right
		if (previous_first_zone != _first_zone) {
			std::rotate(_list_of_single_zone_images.begin(),
			            _list_of_single_zone_images.end() - (previous_first_zone - _first_zone),
			            _list_of_single_zone_images.end());
		}
		keep_zones(previous_first_zone, _last_zone + 1);
		draw_zones(_first_zone, previous_first_zone);
	}
}

int PVParallelView::PVLinesView::update_number_of_zones(int view_x, uint32_t view_width)
{
	size_t old_zones_count = _list_of_zone_width_with_zoom_level.size();
	size_t new_zones_count = get_number_of_managed_zones();
	set_new_view(view_x, view_width);
	_list_of_zone_width_with_zoom_level.resize(new_zones_count);
	return static_cast<int>(new_zones_count) - static_cast<int>(old_zones_count);
}

int PVParallelView::PVLinesView::get_average_zones_width() const
{
	uint32_t sum = 0;

	for (const auto& v : _list_of_zone_width_with_zoom_level) {
		sum += v.get_width();
	}

	return sum / _list_of_zone_width_with_zoom_level.size();
}

void PVParallelView::PVLinesView::reset_zones_width(int wanted_zone_width)
{
	wanted_zone_width = PVCore::clamp<int>(wanted_zone_width, ZoneMinWidth, ZoneMaxWidth);

	const uint32_t lowest_zoom_level = log2(wanted_zone_width / ZoneBaseWidth) * zoom_divisor;
	const uint32_t upper_zoom_level =
	    log2((wanted_zone_width + ZoneBaseWidth) / ZoneBaseWidth) * zoom_divisor;

	const int lowest_width = zoom_level_to_width(lowest_zoom_level);
	const int upper_width = zoom_level_to_width(upper_zoom_level);
	uint32_t zoom_level;

	if (std::abs(wanted_zone_width - lowest_width) < std::abs(upper_width - wanted_zone_width)) {
		zoom_level = lowest_zoom_level;
	} else {
		zoom_level = upper_zoom_level;
	}

	for (ZoneWidthWithZoomLevel& z : _list_of_zone_width_with_zoom_level) {
		z.set_base_zoom_level(zoom_level);
		z.set_base_width(ZoneWidthWithZoomLevel::default_base_width);
	}
}

void PVParallelView::PVLinesView::visit_all_zones_to_render(
    std::function<void(size_t)> const& fzone)
{
	for (size_t z = _first_zone; z < _last_zone + 1; ++z) {
		fzone(z);
	}
}

/******************************************************************************
 ******************************************************************************
 *
 * SingleZoneImages Implementation
 *
 ******************************************************************************
 *****************************************************************************/

void PVParallelView::PVLinesView::SingleZoneImages::cancel_all_and_wait()
{
	// That copy is important if we are multi-threading!
	PVZoneRenderingBCIBase_p zr = last_zr_sel;
	if (zr) {
		zr->cancel();
		zr->wait_end();
		last_zr_sel.reset();
	}

	zr = last_zr_bg;
	if (zr) {
		zr->cancel();
		zr->wait_end();
		last_zr_bg.reset();
	}
}

void PVParallelView::PVLinesView::SingleZoneImages::cancel_last_bg()
{
	// AG: that following copy is *important* !
	PVZoneRenderingBCIBase_p zr = last_zr_bg;
	if (zr) {
		zr->cancel();
	}
}

void PVParallelView::PVLinesView::SingleZoneImages::cancel_last_sel()
{
	// AG: that following copy is *important* !
	PVZoneRenderingBCIBase_p zr = last_zr_sel;
	if (zr) {
		zr->cancel();
	}
}

void PVParallelView::PVLinesView::SingleZoneImages::create_image(PVBCIDrawingBackend& backend,
                                                                 uint32_t width)
{
	sel = backend.create_image(width, PARALLELVIEW_ZT_BBITS);
	bg = backend.create_image(width, PARALLELVIEW_ZT_BBITS);
}

void PVParallelView::PVLinesView::SingleZoneImages::set_width(uint32_t width)
{
	sel->set_width(width);
	bg->set_width(width);
}

/******************************************************************************
 ******************************************************************************
 *
 * ZoneWidthWithZoomLevel Implementation
 *
 ******************************************************************************
 *****************************************************************************/

void PVParallelView::PVLinesView::ZoneWidthWithZoomLevel::decrease_zoom_level()
{
	if (_base_zoom_level > min_zoom_level) {
		--_base_zoom_level;
	}
}

uint32_t PVParallelView::PVLinesView::ZoneWidthWithZoomLevel::get_width() const
{
	return zoom_level_to_width(_base_zoom_level, _base_width);
}

void PVParallelView::PVLinesView::ZoneWidthWithZoomLevel::increase_zoom_level()
{
	if (_base_zoom_level < max_zoom_level) {
		++_base_zoom_level;
	}
}

void PVParallelView::PVLinesView::ZoneWidthWithZoomLevel::set_base_width(int16_t base_width)
{
	if ((base_width > 15) && (base_width < 2048)) {
		_base_width = base_width;
	}
}

void PVParallelView::PVLinesView::ZoneWidthWithZoomLevel::set_base_zoom_level(
    int16_t base_zoom_level)
{
	if ((base_zoom_level > -10000) && (base_zoom_level < 10000)) {
		_base_zoom_level = base_zoom_level;
	}
}
