
#include <pvparallelview/PVHitCountViewSelectionRectangle.h>
#include <pvparallelview/PVHitCountView.h>
#include <pvparallelview/PVSelectionGenerator.h>

#include <picviz/PVView.h>

#include <QApplication>

/*****************************************************************************
 * PVParallelView::PVHitCountViewSelectionRectangle::PVHitCountViewSelectionRectangle
 *****************************************************************************/

PVParallelView::PVHitCountViewSelectionRectangle::PVHitCountViewSelectionRectangle(PVHitCountView* hcv) :
	PVSelectionRectangle(hcv->get_scene()),
	_hcv(hcv)
{}

/*****************************************************************************
 * PVParallelView::PVHitCountViewSelectionRectangle::commit
 *****************************************************************************/

void PVParallelView::PVHitCountViewSelectionRectangle::commit(bool use_selection_modifiers)
{
	QRectF r = get_rect();
	Picviz::PVView& view = lib_view();

	unsigned int modifiers = (unsigned int) QApplication::keyboardModifiers();
	modifiers &= ~Qt::KeypadModifier;

	bool use_selectable = true;
	if (use_selection_modifiers
	    &&
	    (modifiers == PVSelectionGenerator::AND_MODIFIER
	     ||
	     modifiers == PVSelectionGenerator::NAND_MODIFIER)) {
		use_selectable = false;
	}

	PVSelectionGenerator::compute_selection_from_hit_count_view_rect(_hcv->get_hit_graph_manager(),
	                                                                 r, _hcv->get_max_count(),
	                                                                 view.get_volatile_selection(),
	                                                                 use_selectable);
	PVSelectionGenerator::process_selection(view.shared_from_this(), use_selection_modifiers);
}

/*****************************************************************************
 * PVParallelView::PVHitCountViewSelectionRectangle::lib_view
 *****************************************************************************/

Picviz::PVView& PVParallelView::PVHitCountViewSelectionRectangle::lib_view()
{
	return _hcv->lib_view();
}
