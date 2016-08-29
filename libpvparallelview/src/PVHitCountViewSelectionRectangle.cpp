/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

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
	Inendi::PVView& view = lib_view();

	unsigned int modifiers = (unsigned int)QApplication::keyboardModifiers();
	modifiers &= ~Qt::KeypadModifier;

	bool use_selectable = true;
	if (use_selection_modifiers && (modifiers == PVSelectionGenerator::AND_MODIFIER ||
	                                modifiers == PVSelectionGenerator::NAND_MODIFIER)) {
		use_selectable = false;
	}

	Inendi::PVSelection sel(lib_view().get_row_count());
	sel.select_none();

	PVSelectionGenerator::compute_selection_from_hit_count_view_rect(
	    _hcv->get_hit_graph_manager(), r, _hcv->get_max_count(), sel, use_selectable);
	PVSelectionGenerator::process_selection(view, sel, use_selection_modifiers);
}

/*****************************************************************************
 * PVParallelView::PVHitCountViewSelectionRectangle::lib_view
 *****************************************************************************/

Inendi::PVView& PVParallelView::PVHitCountViewSelectionRectangle::lib_view()
{
	return _hcv->lib_view();
}
