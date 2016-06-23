/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <math.h>

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVSerializeArchiveOptions.h>
#include <pvkernel/core/PVConfig.h>
#include <pvkernel/core/inendi_bench.h>

#include <inendi/PVPlotted.h>
#include <inendi/PVRoot.h>
#include <inendi/PVSource.h>
#include <inendi/PVView.h>
#include <inendi/PVLayerFilter.h>
#include <inendi/PVMapped.h>
#include <inendi/PVMapping.h>
#include <inendi/PVPlotted.h>
#include <inendi/PVPlotting.h>

#ifdef WITH_MINESET
#include <inendi/PVMineset.h>
#endif

#include <future>

PVCore::PVHSVColor Inendi::PVView::_default_zombie_line_properties(HSV_COLOR_BLACK);

/******************************************************************************
 *
 * Inendi::PVView::PVView
 *
 *****************************************************************************/
Inendi::PVView::PVView(PVPlotted& plotted)
    : PVCore::PVDataTreeChild<PVPlotted, PVView>(plotted)
    , _axes_combination(get_parent<PVSource>().get_format())
    , floating_selection(get_row_count())
    , post_filter_layer("post_filter_layer", get_row_count())
    , layer_stack_output_layer("view_layer_stack_output_layer", get_row_count())
    , output_layer("output_layer", get_row_count())
    , layer_stack(get_row_count())
    , volatile_selection(get_row_count())
    , _rushnraw_parent(&get_parent<PVSource>().get_rushnraw())
    , _view_id(-1)
    , _active_axis(0)
{
	get_parent<PVSource>().add_view(this);

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
	layer_stack.get_layer_n(0).compute_selectable_count();

	_layer_stack_refreshed.emit();

	process_from_layer_stack();
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
	get_parent<Inendi::PVRoot>().correlations().remove(this);

#ifdef WITH_MINESET
	for (const std::string& mineset_dataset : _mineset_datasets) {
		std::thread req(Inendi::PVMineset::delete_dataset, mineset_dataset);
		req.detach();
	}
#endif

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

void Inendi::PVView::commit_volatile_in_floating_selection()
{
	switch (_state_machine.get_square_area_mode()) {
	case Inendi::PVStateMachine::AREA_MODE_ADD_VOLATILE:
		floating_selection |= volatile_selection;
		break;

	case Inendi::PVStateMachine::AREA_MODE_INTERSECT_VOLATILE:
		floating_selection &= volatile_selection;
		break;

	case Inendi::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE:
		floating_selection = volatile_selection;
		break;

	case Inendi::PVStateMachine::AREA_MODE_SUBSTRACT_VOLATILE:
		floating_selection -= volatile_selection;
		break;

	case Inendi::PVStateMachine::AREA_MODE_OFF:;
	}
}

/******************************************************************************
 *
 * Inendi::PVView::get_axes_count
 *
 *****************************************************************************/
PVCol Inendi::PVView::get_axes_count() const
{
	return _axes_combination.get_axes_count();
}

/******************************************************************************
 *
 * Inendi::PVView::get_axes_names_list
 *
 *****************************************************************************/
QStringList Inendi::PVView::get_axes_names_list() const
{
	return _axes_combination.get_axes_names_list();
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

Inendi::PVAxis const& Inendi::PVView::get_axis(PVCol index) const
{
	return _axes_combination.get_axis(index);
}

Inendi::PVAxis const& Inendi::PVView::get_axis_by_id(axes_comb_id_t const axes_comb_id) const
{
	return _axes_combination.get_axis(_axes_combination.get_index_by_id(axes_comb_id));
}

/******************************************************************************
 *
 * Inendi::PVView::get_axis_name
 *
 *****************************************************************************/
const QString& Inendi::PVView::get_axis_name(PVCol index) const
{
	PVAxis const& axis = _axes_combination.get_axis(index);
	return axis.get_name();
}

QString Inendi::PVView::get_axis_type(PVCol index) const
{
	PVAxis const& axis = _axes_combination.get_axis(index);
	return axis.get_type();
}

QString Inendi::PVView::get_original_axis_name(PVCol axis_id) const
{
	PVAxis const& axis = _axes_combination.get_original_axis(axis_id);
	return axis.get_name();
}

QString Inendi::PVView::get_original_axis_type(PVCol axis_id) const
{
	PVAxis const& axis = _axes_combination.get_original_axis(axis_id);
	return axis.get_type();
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
PVCol Inendi::PVView::get_column_count() const
{
	return get_axes_combination().get_axes_count();
}

/******************************************************************************
 *
 * Inendi::PVView::get_data
 *
 *****************************************************************************/
std::string Inendi::PVView::get_data(PVRow row, PVCol column) const
{
	PVCol real_index = _axes_combination.get_axis_column_index_fast(column);

	return get_rushnraw_parent().at_string(row, real_index);
}

/******************************************************************************
 *
 * Inendi::PVView::get_real_axis_index
 *
 *****************************************************************************/
PVCol Inendi::PVView::get_real_axis_index(PVCol col) const
{
	return _axes_combination.get_axis_column_index(col);
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
 * Inendi::PVView::get_number_of_selected_lines
 *
 *****************************************************************************/
int Inendi::PVView::get_number_of_selected_lines() const
{
	return post_filter_layer.get_selection().bit_count();
}

/******************************************************************************
 *
 * Inendi::PVView::get_original_axes_count
 *
 *****************************************************************************/
PVCol Inendi::PVView::get_original_axes_count() const
{
	return _axes_combination.get_original_axes_count();
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

/******************************************************************************
 *
 * Inendi::PVView::move_active_axis_closest_to_position
 *
 *****************************************************************************/
int Inendi::PVView::move_active_axis_closest_to_position(float x)
{
	/* CODE */
	PVCol new_index = get_active_axis_closest_to_position(x);

	/* We move the axis if there is a movement */
	if (new_index != _active_axis) {
		_axes_combination.move_axis_to_new_position(_active_axis, new_index);
		_active_axis = new_index;

		return 1;
	} else {
		return 0;
	}
}

/******************************************************************************
 *
 * Inendi::PVView::get_active_axis_closest_to_position
 *
 *****************************************************************************/
PVCol Inendi::PVView::get_active_axis_closest_to_position(float x)
{
	PVCol axes_count = _axes_combination.get_axes_count();
	int ret = (int)floor(x + 0.5);
	if (ret < 0) {
		/* We set the leftmost AXIS as destination */
		return 0;
	} else if (ret >= axes_count) {
		/* We set the rightmost AXIS as destination */
		return (PVCol)(axes_count - 1);
	}

	return (PVCol)ret;
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
 * Inendi::PVView::process_from_post_filter_layer
 *
 *****************************************************************************/
void Inendi::PVView::process_from_selection()
{
	process_post_filter_layer();
	process_output_layer();
	process_correlation();
}

/******************************************************************************
 *
 * Inendi::PVView::process_from_layer_stack
 *
 *****************************************************************************/
void Inendi::PVView::process_from_layer_stack()
{
	/* We start by reprocessing the layer_stack */
	process_layer_stack();
	process_post_filter_layer();
	process_output_layer();
	process_correlation();
}

void Inendi::PVView::process_real_output_selection()
{
	// AG: TODO: should be optimised to only create real_output_selection
	process_from_selection();
}

/******************************************************************************
 *
 * Inendi::PVView::process_layer_stack
 *
 *****************************************************************************/
void Inendi::PVView::process_layer_stack()
{
	layer_stack.process(layer_stack_output_layer, get_row_count());
	_update_layer_stack_output_layer.emit();
}

/******************************************************************************
 *
 * Inendi::PVView::process_post_filter_layer
 *
 *****************************************************************************/
void Inendi::PVView::process_post_filter_layer()
{
	/* We treat the selection according to specific SQUARE_AREA_SELECTION mode */
	switch (_state_machine.get_square_area_mode()) {
	case Inendi::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE:
		post_filter_layer.get_selection() = volatile_selection;
		break;

	case Inendi::PVStateMachine::AREA_MODE_ADD_VOLATILE:
		post_filter_layer.get_selection() = floating_selection | volatile_selection;
		break;

	case Inendi::PVStateMachine::AREA_MODE_SUBSTRACT_VOLATILE:
		post_filter_layer.get_selection().AB_sub(floating_selection, volatile_selection);
		break;

	case Inendi::PVStateMachine::AREA_MODE_INTERSECT_VOLATILE:
		post_filter_layer.get_selection() = floating_selection & volatile_selection;
		break;

	default:
		post_filter_layer.get_selection() = layer_stack_output_layer.get_selection();
		break;
	}

	/* We cut the resulting selection with what is available in the layer_stack */
	/* We are in ALL edit_mode */
	post_filter_layer.get_selection() &= layer_stack_output_layer.get_selection();

	/* We simply copy the lines_properties */
	post_filter_layer.get_lines_properties() = layer_stack_output_layer.get_lines_properties();

	_update_output_selection.emit();
}

/******************************************************************************
 *
 * Inendi::PVView::process_output_layer
 *
 *****************************************************************************/
void Inendi::PVView::process_output_layer()
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
			output_layer.get_selection().select_all();
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
		}
	}

	_update_output_layer.emit();
}

/******************************************************************************
 *
 * Inendi::PVView::set_active_axis_closest_to_position
 *
 *****************************************************************************/
void Inendi::PVView::set_active_axis_closest_to_position(float x)
{
	/* VARIABLES */
	int closest_int;
	int axes_count;

	/* CODE */
	closest_int = (int)floor(x + 0.5);
	if (closest_int < 0) {
		/* We set the leftmost AXIS as active */
		_active_axis = 0;
	} else {
		axes_count = _axes_combination.get_axes_count();
		if (closest_int >= axes_count) {
			/* We set the rightmost AXIS as active */
			_active_axis = axes_count - 1;
		} else {
			/* we can safely set the active AXIS to the closest_int */
			_active_axis = closest_int;
		}
	}
}

/******************************************************************************
 *
 * Inendi::PVView::set_axis_name
 *
 *****************************************************************************/
void Inendi::PVView::set_axis_name(PVCol index, const QString& name_)
{
	PVAxis axis;

	_axes_combination.set_axis_name(index, name_);
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
}

/******************************************************************************
 *
 * Inendi::PVView::set_floating_selection
 *
 *****************************************************************************/
void Inendi::PVView::set_floating_selection(PVSelection& selection)
{
	floating_selection = selection;
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
void Inendi::PVView::set_selection_view(PVSelection const& sel)
{
	_state_machine.set_square_area_mode(Inendi::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE);
	volatile_selection = sel;
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

void Inendi::PVView::select_all_nonzb_lines()
{
	_state_machine.set_square_area_mode(Inendi::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE);
	volatile_selection.select_all();
}

void Inendi::PVView::select_no_line()
{
	// Set square area mode w/ volatile
	_state_machine.set_square_area_mode(Inendi::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE);
	volatile_selection.select_none();
}

void Inendi::PVView::select_inv_lines()
{
	// Commit current volatile selection
	commit_volatile_in_floating_selection();
	// Set square area mode w/ volatile
	_state_machine.set_square_area_mode(Inendi::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE);
	floating_selection.select_inverse();
	std::swap(volatile_selection, floating_selection);
}

std::string Inendi::PVView::get_name() const
{
	return std::to_string(get_display_view_id()) + " (" + get_parent<PVMapped>().get_name() + "/" +
	       get_parent<PVPlotted>().get_name() + ")";
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
	_toggle_unselected.emit();
}

void Inendi::PVView::toggle_listing_zombie_visibility()
{
	_state_machine.toggle_listing_zombie_visibility();
	_toggle_zombie.emit();
}

void Inendi::PVView::toggle_view_unselected_zombie_visibility()
{
	_state_machine.toggle_view_unselected_zombie_visibility();
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

void Inendi::PVView::finish_process_from_rush_pipeline()
{
	layer_stack.compute_selectable_count();
}

void Inendi::PVView::set_axes_combination_list_id(PVAxesCombination::columns_indexes_t const& idxes,
                                                  PVAxesCombination::list_axes_t const& axes)
{
	_axis_combination_about_to_update.emit();

	_axes_combination.set_axes_index_list(idxes, axes);

	_axis_combination_updated.emit();
}

PVRow Inendi::PVView::get_plotted_col_min_row(PVCol const combined_col) const
{
	PVCol const col = _axes_combination.get_axis_column_index(combined_col);
	return get_parent<PVPlotted>().get_col_min_row(col);
}

PVRow Inendi::PVView::get_plotted_col_max_row(PVCol const combined_col) const
{
	PVCol const col = _axes_combination.get_axis_column_index(combined_col);
	return get_parent<PVPlotted>().get_col_max_row(col);
}

void Inendi::PVView::sort_indexes(PVCol col,
                                  pvcop::db::indexes& idxes,
                                  tbb::task_group_context* /*ctxt = NULL*/) const
{
	BENCH_START(pvcop_sort);
	pvcop::db::array column = get_rushnraw_parent().collection().column(col);
	idxes.parallel_sort_on(column);
	BENCH_END(pvcop_sort, "pvcop_sort", 0, 0, 1, idxes.size());
}

// Load/save and serialization
void Inendi::PVView::serialize_write(PVCore::PVSerializeObject& so)
{
	so.object("layer-stack", layer_stack, "Layers", true);
	so.object("axes-combination", _axes_combination, "Axes combination", true);
}

void Inendi::PVView::serialize_read(PVCore::PVSerializeObject& so)
{
	so.object("layer-stack", layer_stack, "Layers", true);
	so.object("axes-combination", _axes_combination, "Axes combination", true);
}
