#include <pvparallelview/PVSelectionSquareScatterView.h>
#include <pvparallelview/PVScatterView.h>

#include <cassert>

#include <QApplication>
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

void PVParallelView::PVSelectionSquareScatterView::commit(bool /*use_selection_modifiers*/)
{
	QRectF r = _selection_graphics_item->rect();
	Picviz::PVView& view = lib_view();

	Picviz::PVSelection& sel = view.get_volatile_selection();
	Picviz::PVSelection const& layers_sel = view.get_layer_stack_output_layer().get_selection();

	unsigned int modifiers = (unsigned int) QApplication::keyboardModifiers() & ~Qt::KeypadModifier;
	if (modifiers == (unsigned int) (Qt::ShiftModifier)) {
		PVSelectionGenerator::compute_selection_from_plotted_range(_y2_plotted, _nrows, r.y(), r.y()+r.height(), sel, layers_sel);
	}
	else if (modifiers == (unsigned int) (Qt::ControlModifier)) {
		PVSelectionGenerator::compute_selection_from_plotted_range(_y1_plotted, _nrows, r.x(), r.x()+r.width(), sel, layers_sel);
	}
	else if (modifiers == (unsigned int) (Qt::NoModifier)) {
		PVSelectionGenerator::compute_selection_from_plotteds_ranges(_y1_plotted, _y2_plotted, _nrows, r, sel, layers_sel);
	}

	PVSelectionGenerator::process_selection(view.shared_from_this(), false);
}

Picviz::PVView& PVParallelView::PVSelectionSquareScatterView::lib_view()
{
	return _sv->lib_view();
}
