/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVMineset.h>

#include <inendi/PVAxesCombination.h>   // for PVAxesCombination
#include <inendi/PVCorrelationEngine.h> // for PVCorrelationEngine
#include <inendi/PVLayer.h>             // for PVLayer
#include <inendi/PVLayerFilter.h>       // for PVLayerFilter
#include <inendi/PVLayerStack.h>        // for PVLayerStack
#include <inendi/PVLinesProperties.h>   // for PVLinesProperties
#include <inendi/PVMapped.h>            // for PVMapped
#include <inendi/PVPlotted.h>           // for PVPlotted
#include <inendi/PVRoot.h>              // for PVRoot
#include <inendi/PVSelection.h>         // for PVSelection
#include <inendi/PVSource.h>            // for PVSource
#include <inendi/PVStateMachine.h>      // for PVStateMachine
#include <inendi/PVView.h>              // for PVView, etc

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
#include <pvkernel/core/inendi_bench.h>      // for BENCH_END, BENCH_START

#include <pvbase/types.h> // for PVRow, PVCol

#include <pvcop/collection.h> // for collection
#include <pvcop/db/array.h>   // for indexes, array

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

PVCore::PVHSVColor Inendi::PVView::_default_zombie_line_properties(HSV_COLOR_BLACK);

/******************************************************************************
 *
 * Inendi::PVView::PVView
 *
 *****************************************************************************/
Inendi::PVView::PVView(PVPlotted& plotted)
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
	LIB_CLASS(Inendi::PVLayerFilter)& filters_layer = LIB_CLASS(Inendi::PVLayerFilter)::get();
	LIB_CLASS(Inendi::PVLayerFilter)::list_classes const& lf = filters_layer.get_list();

	for (auto it = lf.begin(); it != lf.end(); it++) {
		filters_args[it->key()] = it->value()->get_default_args_for_view(*this);
	}

	_layer_stack_about_to_refresh.emit();
	PVRow row_count = get_row_count();

	// This function remove all the layers and add the default one with all events
	// selected
	layer_stack.delete_all_layers();
	layer_stack.append_new_layer(row_count, "All events");
	layer_stack.get_layer_n(0).set_lock();
	layer_stack.get_layer_n(0).get_selection() =
	    (const Inendi::PVSelection&)get_parent<PVSource>().get_rushnraw().valid_rows_sel();
	layer_stack.get_layer_n(0).compute_selectable_count();

	_layer_stack_refreshed.emit();

	// does not call ::select_all() to avoid calling uselessly ::process_post_filter_layer()
	_view_selection.select_all();
	process_layer_stack();
}

/******************************************************************************
 *
 * Inendi::PVView::~PVView
 *
 *****************************************************************************/
Inendi::PVView::~PVView()
{
	_about_to_be_delete.emit();
	PVLOG_DEBUG("In PVView destructor: 0x%x\n", this);

	// remove correlation
	get_parent<Inendi::PVRoot>().correlations().remove(this, true /*both_ways*/);

	for (const std::string& mineset_dataset : _mineset_datasets) {
		std::thread req(Inendi::PVMineset::delete_dataset, mineset_dataset);
		req.detach();
	}

	get_parent<PVRoot>().view_being_deleted(this);
}

/******************************************************************************
 *
 * Inendi::PVView::add_new_layer
 *
 *****************************************************************************/
void Inendi::PVView::add_new_layer(QString name)
{
	_layer_stack_about_to_refresh.emit();
	size_t row_count = get_row_count();
	Inendi::PVLayer* layer = layer_stack.append_new_layer(row_count, name);
	layer->compute_selectable_count();

	_layer_stack_refreshed.emit();
	_update_current_min_max.emit();
}

void Inendi::PVView::delete_selected_layer()
{
	_layer_stack_about_to_refresh.emit();
	layer_stack.delete_selected_layer();

	_layer_stack_refreshed.emit();
	_update_current_min_max.emit();
}

void Inendi::PVView::delete_layer_n(int idx)
{
	_layer_stack_about_to_refresh.emit();
	layer_stack.delete_by_index(idx);

	_layer_stack_refreshed.emit();
	_update_current_min_max.emit();
}

void Inendi::PVView::duplicate_selected_layer(const QString& name)
{
	_layer_stack_about_to_refresh.emit();
	PVLayer* new_layer = layer_stack.duplicate_selected_layer(name);
	compute_layer_min_max(*new_layer);
	new_layer->compute_selectable_count();

	_layer_stack_refreshed.emit();
	_update_current_min_max.emit();
}

/******************************************************************************
 * Inendi::PVView::commit_selection_to_layer
 *****************************************************************************/

void Inendi::PVView::commit_selection_to_layer(PVLayer& new_layer)
{
	/* We set it's selection to the final selection */
	new_layer.get_selection() = post_filter_layer.get_selection();
	output_layer.get_lines_properties().A2B_copy_restricted_by_selection(
	    new_layer.get_lines_properties(), new_layer.get_selection());
}

/******************************************************************************
 *
 * Inendi::PVView::get_axes_names_list
 *
 *****************************************************************************/
QStringList Inendi::PVView::get_axes_names_list() const
{
	return _axes_combination.get_combined_names();
}

QStringList Inendi::PVView::get_zones_names_list() const
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

PVRush::PVAxisFormat const& Inendi::PVView::get_axis(PVCombCol index) const
{
	// INFO : It is only to get colors (PVAxisFormat) with index a "combined index"
	return _axes_combination.get_axis(index);
}

/******************************************************************************
 *
 * Inendi::PVView::get_axis_name
 *
 *****************************************************************************/
const QString& Inendi::PVView::get_axis_name(PVCombCol index) const
{
	return _axes_combination.get_axis(index).get_name();
}

QString Inendi::PVView::get_nraw_axis_name(PVCol axis_id) const
{
	return get_parent<Inendi::PVSource>().get_format().get_axes()[axis_id].get_name();
}

// FIXME: This function should be removed
/******************************************************************************
 *
 * Inendi::PVView::get_color_in_output_layer
 *
 *****************************************************************************/
const PVCore::PVHSVColor Inendi::PVView::get_color_in_output_layer(PVRow index) const
{
	return output_layer.get_lines_properties().get_line_properties(index);
}

/******************************************************************************
 *
 * Inendi::PVView::get_column_count
 *
 *****************************************************************************/
PVCombCol Inendi::PVView::get_column_count() const
{
	return PVCombCol(get_axes_combination().get_axes_count());
}

/******************************************************************************
 *
 * Inendi::PVView::get_data
 *
 *****************************************************************************/
std::string Inendi::PVView::get_data(PVRow row, PVCombCol column) const
{
	PVCol real_index = _axes_combination.get_nraw_axis(column);

	return get_rushnraw_parent().at_string(row, real_index);
}

/******************************************************************************
 *
 * Inendi::PVView::get_nraw_axis_index
 *
 *****************************************************************************/
PVCol Inendi::PVView::get_nraw_axis_index(PVCombCol col) const
{
	return _axes_combination.get_nraw_axis(col);
}

/******************************************************************************
 *
 * Inendi::PVView::get_layer_stack
 *
 *****************************************************************************/
Inendi::PVLayerStack& Inendi::PVView::get_layer_stack()
{
	return layer_stack;
}

/******************************************************************************
 *
 * Inendi::PVView::get_layer_stack_layer_n_name
 *
 *****************************************************************************/
QString Inendi::PVView::get_layer_stack_layer_n_name(int n) const
{
	PVLayer const& layer = layer_stack.get_layer_n(n);
	return layer.get_name();
}

/******************************************************************************
 *
 * Inendi::PVView::get_layer_stack_layer_n_visible_state
 *
 *****************************************************************************/
int Inendi::PVView::get_layer_stack_layer_n_visible_state(int n) const
{
	PVLayer const& layer = layer_stack.get_layer_n(n);
	return layer.get_visible();
}

/******************************************************************************
 *
 * Inendi::PVView::get_layer_stack_output_layer
 *
 *****************************************************************************/
Inendi::PVLayer& Inendi::PVView::get_layer_stack_output_layer()
{
	return layer_stack_output_layer;
}

/******************************************************************************
 *
 * Inendi::PVView::get_line_state_in_layer_stack_output_layer
 *
 *****************************************************************************/
bool Inendi::PVView::get_line_state_in_layer_stack_output_layer(PVRow index) const
{
	return layer_stack_output_layer.get_selection().get_line(index);
}

/******************************************************************************
 *
 * Inendi::PVView::get_line_state_in_output_layer
 *
 *****************************************************************************/
bool Inendi::PVView::get_line_state_in_output_layer(PVRow index) const
{
	return output_layer.get_selection().get_line(index);
}

/******************************************************************************
 *
 * Inendi::PVView::get_post_filter_layer
 *
 *****************************************************************************/
Inendi::PVLayer& Inendi::PVView::get_post_filter_layer()
{
	return post_filter_layer;
}

/******************************************************************************
 *
 * Inendi::PVView::get_real_output_selection
 *
 *****************************************************************************/
Inendi::PVSelection const& Inendi::PVView::get_real_output_selection() const
{
	return post_filter_layer.get_selection();
}

/******************************************************************************
 *
 * Inendi::PVView::get_row_count
 *
 *****************************************************************************/
PVRow Inendi::PVView::get_row_count() const
{
	return get_parent<PVSource>().get_row_count();
}

PVRush::PVNraw& Inendi::PVView::get_rushnraw_parent()
{
	return get_parent<PVSource>().get_rushnraw();
}
PVRush::PVNraw const& Inendi::PVView::get_rushnraw_parent() const
{
	return get_parent<PVSource>().get_rushnraw();
}

/******************************************************************************
 *
 * Inendi::PVView::process_correlation
 *
 *****************************************************************************/
void Inendi::PVView::process_correlation()
{
	Inendi::PVRoot& root = get_parent<Inendi::PVRoot>();
	root.process_correlation(this);
}

/******************************************************************************
 *
 * Inendi::PVView::process_layer_stack
 *
 *****************************************************************************/
void Inendi::PVView::process_layer_stack(bool emit_signal)
{
	layer_stack.process(layer_stack_output_layer, get_row_count());

	if (emit_signal) {
		_update_layer_stack_output_layer.emit();
	}

	process_post_filter_layer(emit_signal);
}

/******************************************************************************
 *
 * Inendi::PVView::process_post_filter_layer
 *
 *****************************************************************************/
void Inendi::PVView::process_post_filter_layer(bool emit_signal)
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
 * Inendi::PVView::process_output_layer
 *
 *****************************************************************************/
void Inendi::PVView::process_output_layer(bool emit_signal)
{
	PVLOG_DEBUG("Inendi::PVView::process_output_layer\n");

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
 * Inendi::PVView::set_color_on_active_layer
 *
 *****************************************************************************/
void Inendi::PVView::set_color_on_active_layer(const PVCore::PVHSVColor c)
{
	/* VARIABLES */
	PVLayer& active_layer = layer_stack.get_selected_layer();

	active_layer.get_lines_properties().selection_set_color(get_real_output_selection(), c);
	process_layer_stack();
}

/******************************************************************************
 *
 * Inendi::PVView::set_layer_stack_layer_n_name
 *
 *****************************************************************************/
void Inendi::PVView::set_layer_stack_layer_n_name(int n, QString const& name)
{
	_layer_stack_about_to_refresh.emit();
	PVLayer& layer = layer_stack.get_layer_n(n);
	layer.set_name(name);
	_layer_stack_refreshed.emit();
}

/******************************************************************************
 *
 * Inendi::PVView::set_layer_stack_selected_layer_index
 *
 *****************************************************************************/
void Inendi::PVView::set_layer_stack_selected_layer_index(int index)
{
	_layer_stack_about_to_refresh.emit();
	layer_stack.set_selected_layer_index(index);

	_layer_stack_refreshed.emit();
	_update_current_min_max.emit();
}

/******************************************************************************
 *
 * Inendi::PVView::set_selection_from_layer
 *
 *****************************************************************************/
void Inendi::PVView::set_selection_from_layer(PVLayer const& layer)
{
	set_selection_view(layer.get_selection());
}

/******************************************************************************
 *
 * Inendi::PVView::set_selection_view
 *
 *****************************************************************************/
void Inendi::PVView::set_selection_view(PVSelection const& sel, bool update_ls)
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
 * Inendi::PVView::toggle_layer_stack_layer_n_visible_state
 *
 *****************************************************************************/
void Inendi::PVView::toggle_layer_stack_layer_n_visible_state(int n)
{
	_layer_stack_about_to_refresh.emit();
	PVLayer& layer = layer_stack.get_layer_n(n);

	if (layer.get_visible()) {
		layer.set_visible(0);
	} else {
		layer.set_visible(1);
	}
	_layer_stack_refreshed.emit();
}

/******************************************************************************
 * Inendi::PVView::move_selected_layer_to
 *****************************************************************************/

void Inendi::PVView::move_selected_layer_to(int new_index)
{
	_layer_stack_about_to_refresh.emit();
	get_layer_stack().move_selected_layer_to(new_index);

	_layer_stack_refreshed.emit();
	_update_current_min_max.emit();
}

void Inendi::PVView::select_all()
{
	_view_selection.select_all();
	process_post_filter_layer();
}

void Inendi::PVView::select_none()
{
	_view_selection.select_none();
	process_post_filter_layer();
}

void Inendi::PVView::select_inverse()
{
	_view_selection.select_inverse();
	process_post_filter_layer();
}

std::string Inendi::PVView::get_name() const
{
	if (_name.empty()) {
		return std::to_string(get_display_view_id()) + " (" + get_parent<PVMapped>().get_name() +
		       "/" + get_parent<PVPlotted>().get_name() + ")";
	}
	return _name;
}

void Inendi::PVView::set_name(std::string name)
{
	_name = std::move(name);
}

QString Inendi::PVView::get_window_name() const
{
	QString ret = get_parent<PVSource>().get_window_name() + " | ";
	ret += QString::fromStdString(get_name());
	return ret;
}

Inendi::PVSelection const& Inendi::PVView::get_selection_visible_listing() const
{
	return output_layer.get_selection();
}

void Inendi::PVView::toggle_listing_unselected_visibility()
{
	_state_machine.toggle_listing_unselected_visibility();
	process_output_layer();
	_toggle_unselected.emit();
}

void Inendi::PVView::toggle_listing_zombie_visibility()
{
	_state_machine.toggle_listing_zombie_visibility();
	process_output_layer();
	_toggle_zombie.emit();
}

void Inendi::PVView::toggle_view_unselected_zombie_visibility()
{
	_state_machine.toggle_view_unselected_zombie_visibility();
	process_output_layer();
	_toggle_unselected_zombie_visibility.emit();
}

bool& Inendi::PVView::are_view_unselected_zombie_visible()
{
	return _state_machine.are_view_unselected_zombie_visible();
}

void Inendi::PVView::compute_layer_min_max(Inendi::PVLayer& layer)
{
	layer.compute_min_max(get_parent<Inendi::PVPlotted>());
}

void Inendi::PVView::update_current_layer_min_max()
{
	compute_layer_min_max(get_current_layer());

	_update_current_min_max.emit();
}

void Inendi::PVView::compute_selectable_count(Inendi::PVLayer& layer)
{
	layer.compute_selectable_count();
}

void Inendi::PVView::recompute_all_selectable_count()
{
	layer_stack.compute_selectable_count();
}

void Inendi::PVView::set_axes_combination(std::vector<PVCol> const& comb)
{
	_axis_combination_about_to_update.emit();

	_axes_combination.set_combination(comb);

	_axis_combination_updated.emit();
}

PVRow Inendi::PVView::get_plotted_col_min_row(PVCombCol const combined_col) const
{
	PVCol const col = _axes_combination.get_nraw_axis(combined_col);
	return get_parent<PVPlotted>().get_col_min_row(col);
}

PVRow Inendi::PVView::get_plotted_col_max_row(PVCombCol const combined_col) const
{
	PVCol const col = _axes_combination.get_nraw_axis(combined_col);
	return get_parent<PVPlotted>().get_col_max_row(col);
}

void Inendi::PVView::sort_indexes(PVCol col,
                                  pvcop::db::indexes& idxes,
                                  tbb::task_group_context* /*ctxt = nullptr*/) const
{
	BENCH_START(pvcop_sort);
	const pvcop::db::array& column = get_rushnraw_parent().column(col);
	idxes = column.parallel_sort();
	BENCH_END(pvcop_sort, "pvcop_sort", 0, 0, 1, idxes.size());
}

bool Inendi::PVView::insert_axis(const pvcop::db::type_t& column_type, const pybind11::array& column, const QString& axis_name)
{
	// Insert column in Nraw
	PVRush::PVNraw& nraw = get_rushnraw_parent();
	bool ret = nraw.append_column(column_type, column);

	if (ret) {
		// compute mapping and plotting
		Inendi::PVMapped& mapped = get_parent<PVMapped>();
		mapped.append_column();
		Inendi::PVPlotted& plotted = get_parent<PVPlotted>();
		plotted.append_column();
		mapped.compute();

		// update format
		PVCol col(nraw.column_count()-1);
		PVRush::PVFormat& format = const_cast<PVRush::PVFormat&>(get_parent<PVSource>().get_format()); // FIXME
		PVRush::PVAxisFormat axis_format(col);
		axis_format.set_name(axis_name);
		axis_format.set_type(column_type.c_str());
		axis_format.set_mapping("default"); // FIXME : use string for string
		axis_format.set_plotting("default");
		axis_format.set_color(PVFORMAT_AXIS_COLOR_DEFAULT);
		axis_format.set_titlecolor(PVFORMAT_AXIS_TITLECOLOR_DEFAULT);
		format.insert_axis(axis_format, PVCombCol(0), true); // FIXME
		_axes_combination.axis_append(col);
	}

	return ret;
}

// Load/save and serialization
void Inendi::PVView::serialize_write(PVCore::PVSerializeObject& so) const
{
	so.set_current_status("Saving view...");
	so.set_current_status("Saving layer stack...");
	auto ls_obj = so.create_object("layer-stack");
	layer_stack.serialize_write(*ls_obj);

	so.set_current_status("Saving axes combination...");
	auto ax_comb_obj = so.create_object("axes-combination");
	_axes_combination.serialize_write(*ax_comb_obj);
}

Inendi::PVView& Inendi::PVView::serialize_read(PVCore::PVSerializeObject& so,
                                               Inendi::PVPlotted& parent)
{

	so.set_current_status("Loading view...");
	Inendi::PVView& view = parent.emplace_add_child();

	so.set_current_status("Loading axes combination...");
	auto ax_comb_obj = so.create_object("axes-combination");
	view._axes_combination.set_combination(
	    Inendi::PVAxesCombination::serialize_read(
	        *ax_comb_obj, parent.get_parent<Inendi::PVSource>().get_format())
	        .get_combination());

	so.set_current_status("Loading layer stack...");
	auto ls_obj = so.create_object("layer-stack");
	view.layer_stack = Inendi::PVLayerStack::serialize_read(*ls_obj);

	so.set_current_status("Processing layer stack...");

	/* as PVView' constructor has already reset _view_selection to "all", just need to rebuild
	 * the layer-stack.
	 */
	view.process_layer_stack();

	return view;
}
