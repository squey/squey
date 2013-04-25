#include <pvparallelview/PVSelectionSquareScatterView.h>
#include <pvparallelview/PVScatterView.h>

#include <cassert>

#include <QGraphicsView>

PVParallelView::PVSelectionSquareScatterView::PVSelectionSquareScatterView(PVZoneTree const& zt, PVScatterView* sv):
	PVSelectionSquare(sv->get_scene()),
	_zt(zt),
	_sv(sv)
{
}

void PVParallelView::PVSelectionSquareScatterView::commit(bool use_selection_modifiers)
{
	QRectF r = _selection_graphics_item->rect();
	Picviz::PVView& view = lib_view();
	PVSelectionGenerator::compute_selection_from_scatter_view_rect(_zt, r, view.get_volatile_selection());
	PVSelectionGenerator::process_selection(view.shared_from_this(), use_selection_modifiers);
}

Picviz::PVView& PVParallelView::PVSelectionSquareScatterView::lib_view()
{
	return _sv->lib_view();
}
