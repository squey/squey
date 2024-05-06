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

#include <pvparallelview/PVScatterViewSelectionRectangle.h>
#include <pvparallelview/PVScatterView.h>
#include <pvparallelview/PVSelectionGenerator.h>

#include <squey/PVView.h>

#include <cassert>

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
 * PVParallelView::PVScatterViewSelectionRectangle::PVScatterViewSelectionRectangle
 *****************************************************************************/

PVParallelView::PVScatterViewSelectionRectangle::PVScatterViewSelectionRectangle(
    PVParallelView::PVScatterView* sv)
    : PVSelectionRectangle(sv->get_scene())
    , _y1_scaled(nullptr)
    , _y2_scaled(nullptr)
    , _nrows(0)
    , _sv(sv)
{
}

/*****************************************************************************
 * PVParallelView::PVScatterViewSelectionRectangle::set_scaleds
 *****************************************************************************/

void PVParallelView::PVScatterViewSelectionRectangle::set_scaleds(const uint32_t* y1_scaled,
                                                                   const uint32_t* y2_scaled,
                                                                   const PVRow nrows)
{
	_y1_scaled = y1_scaled;
	_y2_scaled = y2_scaled;
	_nrows = nrows;
}

/*****************************************************************************
 * PVParallelView::PVScatterViewSelectionRectangle::commit
 *****************************************************************************/

void PVParallelView::PVScatterViewSelectionRectangle::commit(bool use_selection_modifiers)
{
	QRectF r = get_rect();

	// Hack to revert scaling inversion
	uint32_t x_min = PVCore::invert_scaling_value(r.x() + r.width());
	uint32_t x_max = PVCore::invert_scaling_value(r.x());
	qreal x_delta = (qreal)x_max - (qreal)x_min;
	r.setX(x_min);
	r.setWidth(x_delta);

	Squey::PVView& view = _sv->lib_view();

	Squey::PVSelection sel(view.get_row_count());
	sel.select_none();
	Squey::PVSelection const& layers_sel = view.get_layer_stack_output_layer().get_selection();

	if (selection_mode() == SelectionMode::VERTICAL) {
		PVSelectionGenerator::compute_selection_from_scaled_range(
		    _y1_scaled, _nrows, std::max(0.0, r.x()), std::max(0.0, r.x() + r.width()), sel,
		    layers_sel);
	} else if (selection_mode() == SelectionMode::HORIZONTAL) {
		PVSelectionGenerator::compute_selection_from_scaled_range(
		    _y2_scaled, _nrows, std::max(0.0, r.y()), std::max(0.0, r.y() + r.height()), sel,
		    layers_sel);
	} else if (selection_mode() == SelectionMode::RECTANGLE) {
		PVSelectionGenerator::compute_selection_from_scaleds_ranges(_y1_scaled, _y2_scaled,
		                                                             _nrows, r, sel, layers_sel);
	} else {
		assert(false);
	}

	PVSelectionGenerator::process_selection(view, sel, use_selection_modifiers);
}
