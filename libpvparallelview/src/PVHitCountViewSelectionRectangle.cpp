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

#include <pvparallelview/PVHitCountViewSelectionRectangle.h>
#include <pvparallelview/PVHitCountView.h>
#include <pvparallelview/PVSelectionGenerator.h>

#include <inendi/PVView.h>

#include <QApplication>

/*****************************************************************************
 * PVParallelView::PVHitCountViewSelectionRectangle::PVHitCountViewSelectionRectangle
 *****************************************************************************/

PVParallelView::PVHitCountViewSelectionRectangle::PVHitCountViewSelectionRectangle(
    PVHitCountView* hcv)
    : PVSelectionRectangle(hcv->get_scene()), _hcv(hcv)
{
}

/*****************************************************************************
 * PVParallelView::PVHitCountViewSelectionRectangle::commit
 *****************************************************************************/

void PVParallelView::PVHitCountViewSelectionRectangle::commit(bool use_selection_modifiers)
{
	QRectF r = get_rect();
	Inendi::PVView& view = _hcv->lib_view();

	unsigned int modifiers = (unsigned int)QApplication::keyboardModifiers();
	modifiers &= ~Qt::KeypadModifier;

	bool use_selectable = true;
	if (use_selection_modifiers && (modifiers == PVSelectionGenerator::AND_MODIFIER ||
	                                modifiers == PVSelectionGenerator::NAND_MODIFIER)) {
		use_selectable = false;
	}

	Inendi::PVSelection sel(view.get_row_count());
	sel.select_none();

	if (_hcv->is_backend_valid()) {
		PVSelectionGenerator::compute_selection_from_hit_count_view_rect(
			_hcv->get_hit_graph_manager(), r, _hcv->get_max_count(), sel, use_selectable);
		PVSelectionGenerator::process_selection(view, sel, use_selection_modifiers);
	}
}
