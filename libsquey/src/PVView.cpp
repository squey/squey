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

#include <squey/PVAxesCombination.h>   // for PVAxesCombination
#include <squey/PVCorrelationEngine.h> // for PVCorrelationEngine
#include <squey/PVLayer.h>             // for PVLayer
#include <squey/PVLayerFilter.h>       // for PVLayerFilter
#include <squey/PVLayerStack.h>        // for PVLayerStack
#include <squey/PVLinesProperties.h>   // for PVLinesProperties
#include <squey/PVMapped.h>            // for PVMapped
#include <squey/PVPlotted.h>           // for PVPlotted
#include <squey/PVRoot.h>              // for PVRoot
#include <squey/PVSelection.h>         // for PVSelection
#include <squey/PVSource.h>            // for PVSource
#include <squey/PVStateMachine.h>      // for PVStateMachine
#include <squey/PVView.h>              // for PVView, etc

#include <pvkernel/rush/PVAxisFormat.h> // for PVAxisFormat
#include <pvkernel/rush/PVFormat.h>     // for PVFormat
#include <pvkernel/rush/PVNraw.h>       // for PVNraw

#include <pvkernel/filter/PVFilterFunction.h>

#include <pvkernel/core/PVArgument.h>        // for PVArgumentList
#include <pvkernel/core/PVClassLibrary.h>    // for LIB_CLASS, etc
#include <pvkernel/core/PVDataTreeObject.h>  // for PVDataTreeChild
#include <pvkernel/core/PVHSVColor.h>        // for PVHSVColor, HSV_COLOR_BLACK
#include <pvkernel/core/PVLogger.h>          // for PVLOG_DEBUG
#include <pvkernel/core/PVOrderedMap.h>      // for PVOrderedMapNode
#include <pvkernel/core/PVSerializeObject.h> // for PVSerializeObject
#include <pvkernel/core/squey_bench.h>      // for BENCH_END, BENCH_START

#include <pvbase/types.h> // for PVRow, PVCol

#include <pvcop/collection.h> // for collection
#include <pvcop/db/array.h>   // for indexes, array

#include <QApplication>
#include <QList>       // for QList
#include <QString>     // for QString, operator+
#include <QStringList> // for QStringList

#include <sigc++/signal.h>  // for signal
#include <tbb/tick_count.h> // for tick_count

#include <cstddef> // for size_t
#include <memory>  // for allocator, __shared_ptr, etc
#include <string>  // for operator+, basic_string, etc
#include <vector>  // for vector

namespace tbb
{
class task_group_context;
} // namespace tbb

PVCore::PVHSVColor Squey::PVView::_default_zombie_line_properties(HSV_COLOR_BLACK);

/******************************************************************************
 *
 * Squey::PVView::PVView
 *
 *****************************************************************************/
Squey::PVView::PVView(PVPlotted& plotted)
    : PVCore::PVDataTreeChild<PVPlotted, PVView>(plotted)
    , _view_selection(get_row_count())
    , post_filter_layer("post_filter_layer", get_row_count())
    , layer_stack_output_layer("view_layer_stack_output_layer", get_row_count())
    , output_layer("output_layer", get_row_count())
    , layer_stack()
    , _axes_combination(get_parent<PVSource>().get_format())
    , _view_id(get_parent<PVRoot>().get_new_view_id())
    , _active_axis(0)
    , _color(get_parent<PVRoot>().get_new_view_color())
{
	get_parent<PVRoot>().select_view(*this);

	// Create layer filter arguments for that view
	LIB_CLASS(Squey::PVLayerFilter)& filters_layer = LIB_CLASS(Squey::PVLayerFilter)::get();
	LIB_CLASS(Squey::PVLayerFilter)::list_classes const& lf = filters_layer.get_list();

	for (const auto & it : lf) {
		filters_args[it.key()] = it.value()->get_default_args_for_view(*this);
	}

	_layer_stack_about_to_refresh.emit();
	PVRow row_count = get_row_count();

	// This function remove all the layers and add the default one with all events
	// selected
	layer_stack.delete_all_layers();
	layer_stack.append_new_layer(row_count, "All events");
	layer_stack.get_layer_n(0).set_lock();
	layer_stack.get_layer_n(0).get_selection() =
	    (const Squey::PVSelection&)get_parent<PVSource>().get_rushnraw().valid_rows_sel();
	layer_stack.get_layer_n(0).compute_selectable_count();

	_layer_stack_refreshed.emit();

	// does not call ::select_all() to avoid calling uselessly ::process_post_filter_layer()
	_view_selection.select_all();
	process_layer_stack();
}

/******************************************************************************
 *
 * Squey::PVView::~PVView
 *
 *****************************************************************************/
Squey::PVView::~PVView()
{
	_about_to_be_delete.emit();
	PVLOG_DEBUG("In PVView destructor: 0x%x\n", this);

	// remove correlation
	get_parent<Squey::PVRoot>().correlations().remove(this, true /*both_ways*/);

	get_parent<PVRoot>().view_being_deleted(this);
}

/******************************************************************************
 *
 * Squey::PVView::add_new_layer
 *
 *****************************************************************************/
void Squey::PVView::add_new_layer(QString name)
{
	_layer_stack_about_to_refresh.emit();
	size_t row_count = get_row_count();
	Squey::PVLayer* layer = layer_stack.append_new_layer(row_count, name);
	layer->compute_selectable_count();

	_layer_stack_refreshed.emit();
	_update_current_min_max.emit();
}

void Squey::PVView::delete_selected_layer()
{
	_layer_stack_about_to_refresh.emit();
	layer_stack.delete_selected_layer();

	_layer_stack_refreshed.emit();
	_update_current_min_max.emit();
}

void Squey::PVView::delete_layer_n(int idx)
{
	_layer_stack_about_to_refresh.emit();
	layer_stack.delete_by_index(idx);

	_layer_stack_refreshed.emit();
	_update_current_min_max.emit();
}

void Squey::PVView::duplicate_selected_layer(const QString& name)
{
	_layer_stack_about_to_refresh.emit();
	PVLayer* new_layer = layer_stack.duplicate_selected_layer(name);
	compute_layer_min_max(*new_layer);
	new_layer->compute_selectable_count();

	_layer_stack_refreshed.emit();
	_update_current_min_max.emit();
}

/******************************************************************************
 * Squey::PVView::commit_selection_to_layer
 *****************************************************************************/

void Squey::PVView::commit_selection_to_layer(PVLayer& new_layer)
{
	/* We set it's selection to the final selection */
	new_layer.get_selection() = post_filter_layer.get_selection();
	output_layer.get_lines_properties().A2B_copy_restricted_by_selection(
	    new_layer.get_lines_properties(), new_layer.get_selection());
}

/******************************************************************************
 * Squey::PVView::commit_selection_to_new_layer
 *****************************************************************************/

void Squey::PVView::commit_selection_to_new_layer(const QString& layer_name, bool should_hide_layers /* = true */)
{
	if (should_hide_layers) {
		hide_layers();
	}

	add_new_layer(layer_name);
	Squey::PVLayer& layer = get_current_layer();

	// We need to configure the layer
	commit_selection_to_layer(layer);
	update_current_layer_min_max();
	compute_selectable_count(layer);
	process_layer_stack();
}

/******************************************************************************
 *
 * Squey::PVView::get_axes_names_list
 *
 *****************************************************************************/
QStringList Squey::PVView::get_axes_names_list() const
{
	return _axes_combination.get_combined_names();
}

QStringList Squey::PVView::get_zones_names_list() const
{
	const QStringList axes = get_axes_names_list();

	QStringList ret;
	ret.reserve(axes.size() - 1);

	const QString del(" <-> ");

	for (int i = 0; i < axes.size() - 1; i++) {
		ret << axes[i] + del + axes[i + 1];
	}

	return ret;
}

PVRush::PVAxisFormat const& Squey::PVView::get_axis(PVCombCol index) const
{
	// INFO : It is only to get colors (PVAxisFormat) with index a "combined index"
	return _axes_combination.get_axis(index);
}

/******************************************************************************
 *
 * Squey::PVView::get_axis_name
 *
 *****************************************************************************/
const QString& Squey::PVView::get_axis_name(PVCombCol index) const
{
	return _axes_combination.get_axis(index).get_name();
}

QString Squey::PVView::get_nraw_axis_name(PVCol axis_id) const
{
	return get_parent<Squey::PVSource>().get_format().get_axes()[axis_id].get_name();
}

// FIXME: This function should be removed
/******************************************************************************
 *
 * Squey::PVView::get_color_in_output_layer
 *
 *****************************************************************************/
const PVCore::PVHSVColor Squey::PVView::get_color_in_output_layer(PVRow index) const
{
	return output_layer.get_lines_properties().get_line_properties(index);
}

/******************************************************************************
 *
 * Squey::PVView::get_column_count
 *
 *****************************************************************************/
PVCombCol Squey::PVView::get_column_count() const
{
	return PVCombCol(get_axes_combination().get_axes_count());
}

/******************************************************************************
 *
 * Squey::PVView::get_data
 *
 *****************************************************************************/
std::string Squey::PVView::get_data(PVRow row, PVCombCol column) const
{
	PVCol real_index = _axes_combination.get_nraw_axis(column);

	return get_rushnraw_parent().at_string(row, real_index);
}

/******************************************************************************
 *
 * Squey::PVView::get_nraw_axis_index
 *
 *****************************************************************************/
PVCol Squey::PVView::get_nraw_axis_index(PVCombCol col) const
{
	return _axes_combination.get_nraw_axis(col);
}

/******************************************************************************
 *
 * Squey::PVView::get_layer_stack
 *
 *****************************************************************************/
Squey::PVLayerStack& Squey::PVView::get_layer_stack()
{
	return layer_stack;
}

/******************************************************************************
 *
 * Squey::PVView::get_layer_stack_layer_n_name
 *
 *****************************************************************************/
QString Squey::PVView::get_layer_stack_layer_n_name(int n) const
{
	PVLayer const& layer = layer_stack.get_layer_n(n);
	return layer.get_name();
}

/******************************************************************************
 *
 * Squey::PVView::get_layer_stack_layer_n_visible_state
 *
 *****************************************************************************/
int Squey::PVView::get_layer_stack_layer_n_visible_state(int n) const
{
	PVLayer const& layer = layer_stack.get_layer_n(n);
	return layer.get_visible();
}

/******************************************************************************
 *
 * Squey::PVView::get_layer_stack_output_layer
 *
 *****************************************************************************/
Squey::PVLayer& Squey::PVView::get_layer_stack_output_layer()
{
	return layer_stack_output_layer;
}

/******************************************************************************
 *
 * Squey::PVView::get_line_state_in_layer_stack_output_layer
 *
 *****************************************************************************/
bool Squey::PVView::get_line_state_in_layer_stack_output_layer(PVRow index) const
{
	return layer_stack_output_layer.get_selection().get_line(index);
}

/******************************************************************************
 *
 * Squey::PVView::get_line_state_in_output_layer
 *
 *****************************************************************************/
bool Squey::PVView::get_line_state_in_output_layer(PVRow index) const
{
	return output_layer.get_selection().get_line(index);
}

/******************************************************************************
 *
 * Squey::PVView::get_post_filter_layer
 *
 *****************************************************************************/
Squey::PVLayer& Squey::PVView::get_post_filter_layer()
{
	return post_filter_layer;
}

/******************************************************************************
 *
 * Squey::PVView::get_real_output_selection
 *
 *****************************************************************************/
Squey::PVSelection const& Squey::PVView::get_real_output_selection() const
{
	return post_filter_layer.get_selection();
}

/******************************************************************************
 *
 * Squey::PVView::get_row_count
 *
 *****************************************************************************/
PVRow Squey::PVView::get_row_count() const
{
	return get_parent<PVSource>().get_row_count();
}

PVRush::PVNraw& Squey::PVView::get_rushnraw_parent()
{
	return get_parent<PVSource>().get_rushnraw();
}
PVRush::PVNraw const& Squey::PVView::get_rushnraw_parent() const
{
	return get_parent<PVSource>().get_rushnraw();
}

/******************************************************************************
 *
 * Squey::PVView::process_correlation
 *
 *****************************************************************************/
void Squey::PVView::process_correlation()
{
	auto& root = get_parent<Squey::PVRoot>();
	root.process_correlation(this);
}

/******************************************************************************
 *
 * Squey::PVView::process_layer_stack
 *
 *****************************************************************************/
void Squey::PVView::process_layer_stack(bool emit_signal)
{
	layer_stack.process(layer_stack_output_layer, get_row_count());

	if (emit_signal) {
		_update_layer_stack_output_layer.emit();
	}

	process_post_filter_layer(emit_signal);
}

/******************************************************************************
 *
 * Squey::PVView::process_post_filter_layer
 *
 *****************************************************************************/
void Squey::PVView::process_post_filter_layer(bool emit_signal)
{
	/* Updating the post_filter_layer selection */
	post_filter_layer.get_selection().inplace_and(layer_stack_output_layer.get_selection(),
	                                              _view_selection);

	/* We simply copy the lines_properties */
	post_filter_layer.get_lines_properties() = layer_stack_output_layer.get_lines_properties();

	if (emit_signal) {
		_update_output_selection.emit();
	}

	process_output_layer(emit_signal);
}

/******************************************************************************
 *
 * Squey::PVView::process_output_layer
 *
 *****************************************************************************/
void Squey::PVView::process_output_layer(bool emit_signal)
{
	PVLOG_DEBUG("Squey::PVView::process_output_layer\n");

	PVLinesProperties& out_lps = output_layer.get_lines_properties();
	PVLinesProperties const& post_lps = post_filter_layer.get_lines_properties();
/* We are now able to process the lines_properties */
#pragma omp parallel for schedule(dynamic, 2048)
	for (PVRow i = 0; i < get_row_count(); i++) {
		/* We check if the event is selected at the end of the process */
		if (layer_stack_output_layer.get_selection().get_line(i)) {
			/* It is selected, so we copy its line properties */
			out_lps.set_line_properties(i, post_lps.get_line_properties(i));
		} else {
			/* The event is a zombie one */
			out_lps.set_line_properties(i, _default_zombie_line_properties);
		}
	}

	/* We First need to check if the UNSELECTED lines are visible */
	if (_state_machine.are_listing_unselected_visible()) {
		/* The UNSELECTED are visible */
		/* Now we need to know if the ZOMBIE are visible */
		if (_state_machine.are_listing_zombie_visible()) {
			/* ZOMBIE are visible */
			output_layer.get_selection() = PVSelection(get_rushnraw_parent().valid_rows_sel());
		} else {
			/* Zombie are not visible */
			output_layer.get_selection() = layer_stack_output_layer.get_selection();
		}
	} else {
		/* UNSELECTED lines are invisible */
		output_layer.get_selection() = post_filter_layer.get_selection();
		/* Now we need to know if the ZOMBIE are visible */
		if (_state_machine.are_listing_zombie_visible()) {
			/* ZOMBIE are visible */
			output_layer.get_selection().or_not(layer_stack_output_layer.get_selection());
			output_layer.get_selection() &= PVSelection(get_rushnraw_parent().valid_rows_sel());
		}
	}

	if (emit_signal) {
		_update_output_layer.emit();
	}

	process_correlation();
}

/******************************************************************************
 *
 * Squey::PVView::set_color_on_active_layer
 *
 *****************************************************************************/
void Squey::PVView::set_color_on_active_layer(const PVCore::PVHSVColor c)
{
	/* VARIABLES */
	PVLayer& active_layer = layer_stack.get_selected_layer();

	active_layer.get_lines_properties().selection_set_color(get_real_output_selection(), c);
	process_layer_stack();
}

/******************************************************************************
 *
 * Squey::PVView::set_layer_stack_layer_n_name
 *
 *****************************************************************************/
void Squey::PVView::set_layer_stack_layer_n_name(int n, QString const& name)
{
	_layer_stack_about_to_refresh.emit();
	PVLayer& layer = layer_stack.get_layer_n(n);
	layer.set_name(name);
	_layer_stack_refreshed.emit();
}

/******************************************************************************
 *
 * Squey::PVView::set_layer_stack_selected_layer_index
 *
 *****************************************************************************/
void Squey::PVView::set_layer_stack_selected_layer_index(int index)
{
	_layer_stack_about_to_refresh.emit();
	layer_stack.set_selected_layer_index(index);

	_layer_stack_refreshed.emit();
	_update_current_min_max.emit();
}

/******************************************************************************
 *
 * Squey::PVView::set_selection_from_layer
 *
 *****************************************************************************/
void Squey::PVView::set_selection_from_layer(PVLayer const& layer)
{
	set_selection_view(layer.get_selection());
}

/******************************************************************************
 *
 * Squey::PVView::set_selection_view
 *
 *****************************************************************************/
void Squey::PVView::set_selection_view(PVSelection const& sel, bool update_ls)
{
	_view_selection = sel;

	if (update_ls) {
		process_layer_stack();
	} else {
		process_post_filter_layer();
	}
}

/******************************************************************************
 *
 * Squey::PVView::toggle_layer_stack_layer_n_visible_state
 *
 *****************************************************************************/
void Squey::PVView::toggle_layer_stack_layer_n_visible_state(int n)
{
	_layer_stack_about_to_refresh.emit();
	PVLayer& layer = layer_stack.get_layer_n(n);

	if (layer.get_visible()) {
		layer.set_visible(false);
	} else {
		layer.set_visible(true);
	}
	_layer_stack_refreshed.emit();
}

/******************************************************************************
 * Squey::PVView::move_selected_layer_to
 *****************************************************************************/

void Squey::PVView::move_selected_layer_to(int new_index)
{
	_layer_stack_about_to_refresh.emit();
	get_layer_stack().move_selected_layer_to(new_index);

	_layer_stack_refreshed.emit();
	_update_current_min_max.emit();
}

void Squey::PVView::select_all()
{
	_view_selection.select_all();
	process_post_filter_layer();
}

void Squey::PVView::select_none()
{
	_view_selection.select_none();
	process_post_filter_layer();
}

void Squey::PVView::select_inverse()
{
	_view_selection.select_inverse();
	process_post_filter_layer();
}

std::string Squey::PVView::get_name() const
{
	if (_name.empty()) {
		return std::to_string(get_display_view_id()) + " (" + get_parent<PVMapped>().get_name() +
		       "/" + get_parent<PVPlotted>().get_name() + ")";
	}
	return _name;
}

void Squey::PVView::set_name(std::string name)
{
	_name = std::move(name);
}

QString Squey::PVView::get_window_name() const
{
	QString ret = get_parent<PVSource>().get_window_name() + " | ";
	ret += QString::fromStdString(get_name());
	return ret;
}

Squey::PVSelection const& Squey::PVView::get_selection_visible_listing() const
{
	return output_layer.get_selection();
}

void Squey::PVView::toggle_listing_unselected_visibility()
{
	_state_machine.toggle_listing_unselected_visibility();
	process_output_layer();
	_toggle_unselected.emit();
}

void Squey::PVView::toggle_listing_zombie_visibility()
{
	_state_machine.toggle_listing_zombie_visibility();
	process_output_layer();
	_toggle_zombie.emit();
}

void Squey::PVView::toggle_view_unselected_zombie_visibility()
{
	_state_machine.toggle_view_unselected_zombie_visibility();
	process_output_layer();
	_toggle_unselected_zombie_visibility.emit();
}

bool& Squey::PVView::are_view_unselected_zombie_visible()
{
	return _state_machine.are_view_unselected_zombie_visible();
}

void Squey::PVView::compute_layer_min_max(Squey::PVLayer& layer)
{
	layer.compute_min_max(get_parent<Squey::PVPlotted>());
}

void Squey::PVView::update_current_layer_min_max()
{
	compute_layer_min_max(get_current_layer());

	_update_current_min_max.emit();
}

void Squey::PVView::compute_selectable_count(Squey::PVLayer& layer)
{
	layer.compute_selectable_count();
}

void Squey::PVView::recompute_all_selectable_count()
{
	layer_stack.compute_selectable_count();
}

void Squey::PVView::set_axes_combination(std::vector<PVCol> const& comb)
{
	_axis_combination_about_to_update.emit();

	_axes_combination.set_combination(comb);

	_axis_combination_updated.emit(true);
}

PVRow Squey::PVView::get_plotted_col_min_row(PVCombCol const combined_col) const
{
	PVCol const col = _axes_combination.get_nraw_axis(combined_col);
	return get_parent<PVPlotted>().get_col_min_row(col);
}

PVRow Squey::PVView::get_plotted_col_max_row(PVCombCol const combined_col) const
{
	PVCol const col = _axes_combination.get_nraw_axis(combined_col);
	return get_parent<PVPlotted>().get_col_max_row(col);
}

void Squey::PVView::sort_indexes(PVCol col,
                                  pvcop::db::indexes& idxes,
                                  tbb::task_group_context* /*ctxt = nullptr*/) const
{
	BENCH_START(pvcop_sort);
	const pvcop::db::array& column = get_rushnraw_parent().column(col);
	idxes = column.parallel_sort();
	BENCH_END(pvcop_sort, "pvcop_sort", 0, 0, 1, idxes.size());
}

bool Squey::PVView::insert_axis(const pvcop::db::type_t& column_type, const pybind11::array& column, const QString& axis_name)
{
	// Insert column in Nraw
	PVRush::PVNraw& nraw = get_rushnraw_parent();
	bool ret = nraw.append_column(column_type, column);

	if (ret) {
		// update format
		PVCol col(nraw.column_count()-1);
		auto& format = const_cast<PVRush::PVFormat&>(get_parent<PVSource>().get_format()); // FIXME
		PVRush::PVAxisFormat axis_format(col);
		axis_format.set_name(axis_name);
		axis_format.set_type(column_type.c_str());
		axis_format.set_mapping("default"); // FIXME : use string for string
		axis_format.set_plotting("default");
		axis_format.set_color(PVFORMAT_AXIS_COLOR_DEFAULT);
		axis_format.set_titlecolor(PVFORMAT_AXIS_TITLECOLOR_DEFAULT);
		format.insert_axis(axis_format, PVCombCol(0), true); // FIXME
		_axes_combination.axis_append(col);

		// compute mapping and plotting
		auto& mapped = get_parent<PVMapped>();
		mapped.append_mapped();
		auto& plotted = get_parent<PVPlotted>();
		plotted.append_plotted();
		mapped.compute();
	}

	return ret;
}

void Squey::PVView::delete_axis(PVCombCol comb_col)
{
	// Remove axis (or axes) from axes combination
	PVCol col = _axes_combination.get_nraw_axis(comb_col);

	// Notify axes combination update on Qt GUI thread
    QMetaObject::invokeMethod(qApp, [&,col](){
		// Remove axes
		auto& format = const_cast<PVRush::PVFormat&>(get_parent<PVSource>().get_format()); // FIXME
		auto& original_format = const_cast<PVRush::PVFormat&>(get_parent<PVSource>().get_original_format()); // FIXME
		_axis_combination_about_to_update.emit();
		_axes_combination.delete_axes(col);
		format.delete_axis(col);
		if (format.has_multi_inputs() and col != 0) {
			original_format.delete_axis(col-PVCol(1));
		}
		_axis_combination_updated.emit(true);
    }, Qt::BlockingQueuedConnection);

	// Delete mapping and plotting
	get_parent<PVPlotted>().delete_plotted(col);
	get_parent<PVMapped>().delete_mapped(col);

	// Delete column from disk
	PVRush::PVNraw& nraw = get_rushnraw_parent();
	nraw.delete_column(col);
}

// Load/save and serialization
void Squey::PVView::serialize_write(PVCore::PVSerializeObject& so) const
{
	so.set_current_status("Saving view...");
	so.set_current_status("Saving layer stack...");
	auto ls_obj = so.create_object("layer-stack");
	layer_stack.serialize_write(*ls_obj);

	so.set_current_status("Saving axes combination...");
	auto ax_comb_obj = so.create_object("axes-combination");
	_axes_combination.serialize_write(*ax_comb_obj);
}

Squey::PVView& Squey::PVView::serialize_read(PVCore::PVSerializeObject& so,
                                               Squey::PVPlotted& parent)
{

	so.set_current_status("Loading view...");
	Squey::PVView& view = parent.emplace_add_child();

	so.set_current_status("Loading axes combination...");
	auto ax_comb_obj = so.create_object("axes-combination");
	view._axes_combination.set_combination(
	    Squey::PVAxesCombination::serialize_read(
	        *ax_comb_obj, parent.get_parent<Squey::PVSource>().get_format())
	        .get_combination());

	so.set_current_status("Loading layer stack...");
	auto ls_obj = so.create_object("layer-stack");
	view.layer_stack = Squey::PVLayerStack::serialize_read(*ls_obj);

	so.set_current_status("Processing layer stack...");

	/* as PVView' constructor has already reset _view_selection to "all", just need to rebuild
	 * the layer-stack.
	 */
	view.process_layer_stack();

	return view;
}
