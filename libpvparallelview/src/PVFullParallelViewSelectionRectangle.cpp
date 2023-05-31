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

#include <pvparallelview/PVFullParallelViewSelectionRectangle.h>

#include <pvparallelview/PVFullParallelView.h>
#include <pvparallelview/PVFullParallelScene.h>
#include <pvparallelview/PVLinesView.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVSelectionGenerator.h>

#include <iostream>

#define print_r(R) __print_rect(#R, R)
#define print_rect(R) __print_rect(#R, R)

template <typename R>
void __print_rect(const char* text, const R& r)
{
	std::cout << text << ": " << r.x() << " " << r.y() << ", " << r.width() << " " << r.height()
	          << std::endl;
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewSelectionRectangle::PVFullParallelViewSelectionRectangle
 *****************************************************************************/

PVParallelView::PVFullParallelViewSelectionRectangle::PVFullParallelViewSelectionRectangle(
    PVFullParallelScene* fps)
    : PVSelectionRectangle(fps), _fps(fps)
{
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewSelectionRectangle::clear
 *****************************************************************************/

void PVParallelView::PVFullParallelViewSelectionRectangle::clear()
{
	PVSelectionRectangle::clear();
	_barycenter.clear();
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewSelectionRectangle::updsate_position
 *****************************************************************************/

void PVParallelView::PVFullParallelViewSelectionRectangle::update_position()
{
	size_t zone_index1 = _barycenter.zone_index1;
	size_t zone_index2 = _barycenter.zone_index2;

	if ((zone_index1 == PVZONEINDEX_INVALID) || (zone_index2 == PVZONEINDEX_INVALID)) {
		return;
	}

	auto const& lines_view = get_lines_view();

	if (zone_index1 >= lines_view.get_number_of_managed_zones() ||
	    zone_index2 >= lines_view.get_number_of_managed_zones()) {
		clear();
		return;
	}

	double factor1 = _barycenter.factor1;
	double factor2 = _barycenter.factor2;

	double new_left = lines_view.get_left_border_position_of_zone_in_scene(zone_index1) +
	                  double(lines_view.get_zone_width(zone_index1)) * factor1;
	double new_right = lines_view.get_left_border_position_of_zone_in_scene(zone_index2) +
	                   double(lines_view.get_zone_width(zone_index2)) * factor2;
	double abs_top = get_rect().top();
	double abs_bottom = get_rect().bottom();

	set_rect(QRectF(QPointF(new_left, abs_top), QPointF(new_right, abs_bottom)), false);
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewSelectionRectangle::commit
 *****************************************************************************/

void PVParallelView::PVFullParallelViewSelectionRectangle::commit(bool use_selection_modifiers)
{
	QRectF srect = get_rect();

	// Too much on the left dude!
	if (srect.x() + srect.width() <= 0) {
		return;
	}

	auto const& lines_view = get_lines_view();

	// Too much on the right, stop drinking!
	const int32_t pos_end = lines_view.get_right_border_of_scene();
	if (srect.x() >= pos_end) {
		return;
	}

	const size_t zone_index_start = lines_view.get_zone_index_from_scene_pos(srect.x());
	const size_t zone_index_end =
	    lines_view.get_zone_index_from_scene_pos(srect.x() + srect.width());

	Squey::PVSelection sel(scene_parent()->lib_view().get_row_count());
	sel.select_none();

	const int axis_width = lines_view.get_axis_width();

	for (size_t z = zone_index_start; z <= zone_index_end; z++) {
		QRect r = scene_parent()->map_to_axis(z, srect);
		const int zone_width = lines_view.get_zone_width(z);
		const int rleft = r.left();
		const int rright = r.right();
		r.setLeft(std::clamp(rleft - axis_width, 0, zone_width));
		r.setRight(std::clamp(rright - axis_width, 0, zone_width));
		PVSelectionGenerator::compute_selection_from_parallel_view_rect(
		    zone_width,
		    lines_view.get_zones_manager().get_zone_tree(
		        lines_view.get_zones_manager().get_zone_id(z)),
		    r, sel);
	}

	store();

	PVSelectionGenerator::process_selection(scene_parent()->lib_view(), sel,
	                                        use_selection_modifiers);
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewSelectionRectangle::store
 *****************************************************************************/

void PVParallelView::PVFullParallelViewSelectionRectangle::store()
{
	auto const& lines_view = get_lines_view();
	auto const& scene = *scene_parent();
	const auto axis_width = lines_view.get_axis_width();

	const auto barycenter_store = [axis_width, &lines_view, &scene](
	                                  size_t& zone_index, double& factor, const double abs_pos) {
		zone_index = lines_view.get_zone_index_from_scene_pos(abs_pos);
		const double zone_width = lines_view.get_zone_width(zone_index);
		const double x_pos = scene.map_to_axis(zone_index, QPointF(abs_pos, 0)).x();
		const double alpha = std::max(0., x_pos - axis_width);
		factor = alpha / zone_width;
	};
	barycenter_store(_barycenter.zone_index1, _barycenter.factor1, get_rect().left());
	barycenter_store(_barycenter.zone_index2, _barycenter.factor2, get_rect().right());
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewSelectionRectangle::scene_parent
 *****************************************************************************/

PVParallelView::PVFullParallelScene*
PVParallelView::PVFullParallelViewSelectionRectangle::scene_parent()
{
	return static_cast<PVParallelView::PVFullParallelScene*>(_fps);
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewSelectionRectangle::scene_parent
 *****************************************************************************/

PVParallelView::PVFullParallelScene const*
PVParallelView::PVFullParallelViewSelectionRectangle::scene_parent() const
{
	return static_cast<PVParallelView::PVFullParallelScene*>(_fps);
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewSelectionRectangle::get_lines_view
 *****************************************************************************/

PVParallelView::PVLinesView const&
PVParallelView::PVFullParallelViewSelectionRectangle::get_lines_view() const
{
	return scene_parent()->get_lines_view();
}
