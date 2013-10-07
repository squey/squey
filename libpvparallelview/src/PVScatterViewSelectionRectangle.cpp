
#include <pvparallelview/PVScatterViewSelectionRectangle.h>
#include <pvparallelview/PVScatterView.h>
#include <pvparallelview/PVSelectionGenerator.h>

#include <picviz/PVView.h>

#include <cassert>

#include <iostream>

#define print_r(R) __print_rect(#R, R)
#define print_rect(R) __print_rect(#R, R)

template <typename R>
void __print_rect(const char *text, const R &r)
{
	std::cout << text << ": "
	          << r.x() << " " << r.y() << ", "
	          << r.width() << " " << r.height()
	          << std::endl;
}

/*****************************************************************************
 * PVParallelView::PVScatterViewSelectionRectangle::PVScatterViewSelectionRectangle
 *****************************************************************************/

PVParallelView::PVScatterViewSelectionRectangle::PVScatterViewSelectionRectangle(PVParallelView::PVScatterView* sv) :
	PVSelectionRectangle(sv->get_scene()),
	_y1_plotted(nullptr),
	_y2_plotted(nullptr),
	_nrows(0),
	_sv(sv)
{}

/*****************************************************************************
 * PVParallelView::PVScatterViewSelectionRectangle::set_plotteds
 *****************************************************************************/

void PVParallelView::PVScatterViewSelectionRectangle::set_plotteds(const uint32_t* y1_plotted,
                                                                   const uint32_t* y2_plotted,
                                                                   const PVRow nrows)
{
	_y1_plotted = y1_plotted;
	_y2_plotted = y2_plotted;
	_nrows = nrows;
}

/*****************************************************************************
 * PVParallelView::PVScatterViewSelectionRectangle::commit
 *****************************************************************************/

void PVParallelView::PVScatterViewSelectionRectangle::commit(bool use_selection_modifiers)
{
	QRectF r = get_rect();

	// Hack to revert plotting inversion
	uint32_t x_min = PVCore::invert_plotting_value(r.x()+r.width());
	uint32_t x_max = PVCore::invert_plotting_value(r.x());
	qreal x_delta = (qreal)x_max - (qreal)x_min;
	r.setX(x_min);
	r.setWidth(x_delta);

	Picviz::PVView& view = lib_view();

	Picviz::PVSelection& sel = view.get_volatile_selection();
	Picviz::PVSelection const& layers_sel = view.get_layer_stack_output_layer().get_selection();

	if (selection_mode() == SelectionMode::VERTICAL) {
		PVSelectionGenerator::compute_selection_from_plotted_range(_y1_plotted, _nrows,
		                                                           std::max(0.0, r.x()),
		                                                           std::max(0.0, r.x()+r.width()),
		                                                           sel, layers_sel);
	} else if (selection_mode() == SelectionMode::HORIZONTAL) {
		PVSelectionGenerator::compute_selection_from_plotted_range(_y2_plotted, _nrows,
		                                                           std::max(0.0, r.y()),
		                                                           std::max(0.0, r.y()+r.height()),
		                                                           sel, layers_sel);
	} else if (selection_mode() == SelectionMode::RECTANGLE) {
		PVSelectionGenerator::compute_selection_from_plotteds_ranges(_y1_plotted, _y2_plotted,
		                                                             _nrows,
		                                                             r, sel, layers_sel);
	} else {
		assert(false);
	}

	PVSelectionGenerator::process_selection(view.shared_from_this(), use_selection_modifiers);
}

/*****************************************************************************
 * PVParallelView::PVScatterViewSelectionRectangle::lib_view
 *****************************************************************************/

Picviz::PVView& PVParallelView::PVScatterViewSelectionRectangle::lib_view()
{
	return _sv->lib_view();
}
