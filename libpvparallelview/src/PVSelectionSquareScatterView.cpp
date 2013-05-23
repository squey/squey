#include <pvparallelview/PVSelectionSquareScatterView.h>
#include <pvparallelview/PVScatterView.h>

#include <cassert>

#include <QGraphicsView>

PVParallelView::PVSelectionSquareScatterView::PVSelectionSquareScatterView(
	PVScatterView* sv
) :
	PVSelectionSquare(sv->get_scene()),
	_y1_plotted(nullptr),
	_y2_plotted(nullptr),
	_nrows(0),
	_sv(sv)
{
}

void PVParallelView::PVSelectionSquareScatterView::set_plotteds(const uint32_t* y1_plotted, const uint32_t* y2_plotted, const PVRow nrows)
{
	_y1_plotted = y1_plotted;
	_y2_plotted = y2_plotted;
	_nrows = nrows;
}

void PVParallelView::PVSelectionSquareScatterView::commit(bool use_selection_modifiers)
{
	QRectF r = _selection_graphics_item->rect();
	Picviz::PVView& view = lib_view();
	PVSelectionGenerator::compute_selection_from_scatter_view_rect(_y1_plotted, _y2_plotted, _nrows, r, view.get_volatile_selection(), view.get_layer_stack_output_layer().get_selection());
	PVSelectionGenerator::process_selection(view.shared_from_this(), use_selection_modifiers);
}

Picviz::PVView& PVParallelView::PVSelectionSquareScatterView::lib_view()
{
	return _sv->lib_view();
}
