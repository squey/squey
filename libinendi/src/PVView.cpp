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

#include <tbb/tick_count.h>

PVCore::PVHSVColor Inendi::PVView::_default_zombie_line_properties(HSV_COLOR_BLACK);

/******************************************************************************
 *
 * Inendi::PVView::PVView
 *
 *****************************************************************************/
Inendi::PVView::PVView()
    : post_filter_layer("post_filter_layer")
    , layer_stack_output_layer("view_layer_stack_output_layer")
    , output_layer("output_layer")
    , _is_consistent(false)
    , _rushnraw_parent(nullptr)
    , _view_id(-1)
    , _active_axis(0)
{
	QSettings& pvconfig = PVCore::PVConfig::get().config();
	last_extractor_batch_size =
	    pvconfig.value("pvkernel/rush/extract_next", PVEXTRACT_NUMBER_LINES_NEXT_DEFAULT).toInt();
}

void Inendi::PVView::set_parent_from_ptr(PVPlotted* plotted)
{
	data_tree_view_t::set_parent_from_ptr(plotted);

	_rushnraw_parent = &get_parent<PVSource>()->get_rushnraw();

	// Create layer filter arguments for that view
	LIB_CLASS(Inendi::PVLayerFilter)& filters_layer = LIB_CLASS(Inendi::PVLayerFilter)::get();
	LIB_CLASS(Inendi::PVLayerFilter)::list_classes const& lf = filters_layer.get_list();

	LIB_CLASS(Inendi::PVLayerFilter)::list_classes::const_iterator it;

	for (it = lf.begin(); it != lf.end(); it++) {
		filters_args[it->key()] = it->value()->get_default_args_for_view(*this);
	}

	/**
	 *  Avoid inializing view when default constructed...
	 */
	PVRow row_count = get_row_count();
	if (row_count) {
		set_row_count(row_count);
		reset_view();
	}
}

void Inendi::PVView::process_parent_plotted()
{
	// Init default axes combination from source
	Inendi::PVPlotted* plotted = get_parent();
	PVSource* source = plotted->get_parent<PVSource>();
	_axes_combination.set_from_format(source->get_format());
	_axes_combination.set_axis_name(
	    0, _axes_combination.get_axis(0).get_name()); // Hack to detach QVector

	set_row_count(get_row_count());

	// First process
	// select_all_nonzb_lines(); Fixes bug #279
	nu_selection.select_none();

	process_from_layer_stack();

	_is_consistent = true;
}

void Inendi::PVView::reset_view()
{
	reset_layers();

	PVSource* source = get_parent<PVSource>();
	_axes_combination = source->get_axes_combination();
	_axes_combination.set_axis_name(
	    0, _axes_combination.get_axis(0).get_name()); // Hack to detach QVector
}

void Inendi::PVView::set_fake_axes_comb(PVCol const ncols)
{
	_axes_combination.clear();
	for (PVCol c = 0; c < ncols; c++) {
		PVAxis axis("integer", "default", "port");
		axis.set_name(QString("axis ") + QString::number(c));
		axis.set_titlecolor("#ffffff");
		_axes_combination.axis_append(axis);
	}
}

/******************************************************************************
 *
 * Inendi::PVView::~PVView
 *
 *****************************************************************************/
Inendi::PVView::~PVView()
{
	PVLOG_DEBUG("In PVView destructor: 0x%x\n", this);

#ifdef WITH_MINESET
	for (const std::string& mineset_dataset : _mineset_datasets) {
		std::thread req(Inendi::PVMineset::delete_dataset, mineset_dataset);
		req.detach();
	}
#endif

	PVRoot* root = get_parent<PVRoot>();
	if (root) {
		root->view_being_deleted(this);
	}
}

/******************************************************************************
 *
 * Inendi::PVView::reset_layers
 *
 *****************************************************************************/
void Inendi::PVView::reset_layers()
{
	// FIXME: this is a workaround to have a view without a source working
	// as 'Tpview_zone_tree_dump_load' and 'Tpview_zoomed_zone_tree_dump_load'
	// tests create a PVView without a PVSource as indirect parent...
	if (not get_parent()) {
		return;
	}

	PVRow row_count = get_row_count();

	// This function remove all the layers and add the default one with all events
	// selected
	bool old_consistent = _is_consistent;
	_is_consistent = false;
	layer_stack.delete_all_layers();
	layer_stack.append_new_layer(row_count);
	layer_stack.get_layer_n(0).reset_to_full_and_default_color(row_count);
	if (row_count != 0) {
		/* when a .pvi is loaded, the mapped and the plotted are
		 * uninitialized when the view is created (the rush pipeline
		 * is runned later).
		 */
		layer_stack.get_layer_n(0).compute_selectable_count(row_count);
	}
	post_filter_layer.reset_to_full_and_default_color(row_count);
	layer_stack_output_layer.reset_to_full_and_default_color(row_count);
	output_layer.reset_to_full_and_default_color(row_count);

	_is_consistent = old_consistent;
}

/******************************************************************************
 *
 * Inendi::PVView::add_new_layer
 *
 *****************************************************************************/
void Inendi::PVView::add_new_layer(QString name)
{
	Inendi::PVLayer* layer = layer_stack.append_new_layer(get_row_count(), name);
	layer->compute_selectable_count(get_row_count());
}

void Inendi::PVView::add_new_layer_from_file(const QString& path)
{
	// Create a new layer
	Inendi::PVLayer* layer = layer_stack.append_new_layer(get_row_count());

	// And load it
	layer->load_from_file(path);
	layer->compute_min_max(*get_parent<Inendi::PVPlotted>());
	layer->compute_selectable_count(get_row_count());
}

void Inendi::PVView::delete_selected_layer()
{
	layer_stack.delete_selected_layer();
}

void Inendi::PVView::delete_layer_n(int idx)
{
	layer_stack.delete_by_index(idx);
}

void Inendi::PVView::duplicate_selected_layer(const QString& name)
{
	PVLayer* new_layer = layer_stack.duplicate_selected_layer(name);
	compute_layer_min_max(*new_layer);
	new_layer->compute_selectable_count(get_row_count());
}

void Inendi::PVView::load_from_file(const QString& file)
{
	layer_stack.load_from_file(file);
	layer_stack.compute_min_maxs(*get_parent<Inendi::PVPlotted>());
	layer_stack.compute_selectable_count(get_row_count());
}

/******************************************************************************
 *
 * Inendi::PVView::apply_filter_named_select_all
 *
 *****************************************************************************/
void Inendi::PVView::apply_filter_named_select_all()
{
	post_filter_layer.get_selection().select_all();
}

/******************************************************************************
 * Inendi::PVView::commit_selection_to_layer
 *****************************************************************************/

void Inendi::PVView::commit_selection_to_layer(PVLayer& new_layer)
{
	/* We set it's selection to the final selection */
	new_layer.get_selection() = post_filter_layer.get_selection();
	output_layer.get_lines_properties().A2B_copy_restricted_by_selection_and_nelts(
	    new_layer.get_lines_properties(), new_layer.get_selection(), get_row_count());
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
 * Inendi::PVView::expand_selection_on_axis
 *
 *****************************************************************************/
void Inendi::PVView::expand_selection_on_axis(PVCol axis_id, QString const& mode)
{
	commit_volatile_in_floating_selection();
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
 * Inendi::PVView::get_layer_stack_layer_n_locked_state
 *
 *****************************************************************************/
int Inendi::PVView::get_layer_stack_layer_n_locked_state(int n) const
{
	PVLayer const& layer = layer_stack.get_layer_n(n);
	return layer.get_locked();
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
 * Inendi::PVView::get_nu_selection
 *
 *****************************************************************************/
Inendi::PVSelection& Inendi::PVView::get_nu_selection()
{
	return nu_selection;
}

/******************************************************************************
 *
 * Inendi::PVView::get_number_of_selected_lines
 *
 *****************************************************************************/
int Inendi::PVView::get_number_of_selected_lines() const
{
	return real_output_selection.get_number_of_selected_lines_in_range(0, get_row_count());
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
 * Inendi::PVView::get_output_layer
 *
 *****************************************************************************/
Inendi::PVLayer& Inendi::PVView::get_output_layer()
{
	return output_layer;
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
	return real_output_selection;
}

/******************************************************************************
 *
 * Inendi::PVView::get_row_count
 *
 *****************************************************************************/
PVRow Inendi::PVView::get_row_count() const
{
	return get_parent<PVSource>()->get_row_count();
}

/******************************************************************************
 *
 * Inendi::PVView::set_row_count
 *
 *****************************************************************************/
void Inendi::PVView::set_row_count(PVRow row_count)
{
	layer_stack.set_row_count(row_count);
	floating_selection.set_count(row_count);
	post_filter_layer.set_count(row_count);
	layer_stack_output_layer.set_count(row_count);
	output_layer.set_count(row_count);
	nu_selection.set_count(row_count);
	real_output_selection.set_count(row_count);
	volatile_selection.set_count(row_count);
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
 * Inendi::PVView::process_eventline
 *
 *****************************************************************************/
void Inendi::PVView::process_eventline()
{
	/* We compute the real_output_selection */
	real_output_selection = post_filter_layer.get_selection();

	/* We refresh the nu_selection */
	nu_selection = std::move(~layer_stack_output_layer.get_selection());

	nu_selection |= real_output_selection;

	PVLinesProperties& out_lps = output_layer.get_lines_properties();
	PVLinesProperties const& post_lps = post_filter_layer.get_lines_properties();
	/* We are now able to process the lines_properties */
	for (PVRow i = 0; i < get_row_count(); i++) {
		/* We check if the event is selected at the end of the process */
		PVCore::PVHSVColor& out_lp = out_lps.get_line_properties(i);
		PVCore::PVHSVColor const& post_lp = post_lps.get_line_properties(i);
		if (real_output_selection.get_line(i)) {
			/* It is selected, so we copy its line properties */
			out_lp = post_lp;
		} else {
			/* It is not selected in the end, so we check if it was available in the
			 * beginning */
			if (layer_stack_output_layer.get_selection().get_line(i)) {
				/* The event was available, but is unselected */
				out_lp = post_lp;
			} else {
				/* The event is a zombie one */
				out_lp = _default_zombie_line_properties;
			}
		}
	}
}

/******************************************************************************
 *
 * Inendi::PVView::process_from_eventline
 *
 *****************************************************************************/
void Inendi::PVView::process_from_eventline()
{
	process_eventline();
	process_visibility();
}

/******************************************************************************
 *
 * Inendi::PVView::process_from_layer_stack
 *
 *****************************************************************************/
void Inendi::PVView::process_from_layer_stack()
{
	tbb::tick_count start = tbb::tick_count::now();

	/* We start by reprocessing the layer_stack */
	process_layer_stack();
	process_selection();
	process_eventline();
	process_visibility();

	tbb::tick_count end = tbb::tick_count::now();
	PVLOG_INFO("(Inendi::PVView::process_from_layer_stack) function took %0.4f "
	           "seconds.\n",
	           (end - start).seconds());
}

/******************************************************************************
 *
 * Inendi::PVView::process_from_selection
 *
 *****************************************************************************/
void Inendi::PVView::process_from_selection()
{
	PVLOG_DEBUG("Inendi::PVView::%s\n", __FUNCTION__);
	process_selection();
	process_eventline();
	process_visibility();
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
}

/******************************************************************************
 *
 * Inendi::PVView::process_selection
 *
 *****************************************************************************/
void Inendi::PVView::process_selection()
{
	/* We treat the selection according to specific SQUARE_AREA_SELECTION mode */
	switch (_state_machine.get_square_area_mode()) {
	case Inendi::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE:
		post_filter_layer.get_selection() = volatile_selection;
		break;

	case Inendi::PVStateMachine::AREA_MODE_ADD_VOLATILE:
		post_filter_layer.get_selection() = std::move(floating_selection | volatile_selection);
		break;

	case Inendi::PVStateMachine::AREA_MODE_SUBSTRACT_VOLATILE:
		post_filter_layer.get_selection().AB_sub(floating_selection, volatile_selection);
		break;

	case Inendi::PVStateMachine::AREA_MODE_INTERSECT_VOLATILE:
		post_filter_layer.get_selection() = std::move(floating_selection & volatile_selection);
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

	/* Now we MUST refresh the index_array associated to nznu */
	/* WARNING ! nothing is done here because this function should be followed by
	 * process_eventline which changes the selection and DO update the
	 * nznu_index_array */
}

/******************************************************************************
 *
 * Inendi::PVView::process_visibility
 *
 *****************************************************************************/
void Inendi::PVView::process_visibility()
{
	PVLOG_DEBUG("Inendi::PVView::process_visibility\n");
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
		output_layer.get_selection() = real_output_selection;
		/* Now we need to know if the ZOMBIE are visible */
		if (_state_machine.are_listing_zombie_visible()) {
			/* ZOMBIE are visible */
			output_layer.get_selection().or_not(layer_stack_output_layer.get_selection());
		}
	}
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

	active_layer.get_lines_properties().selection_set_color(get_real_output_selection(),
	                                                        get_row_count(), c);
}

/******************************************************************************
 *
 * Inendi::PVView::set_color_on_post_filter_layer
 *
 *****************************************************************************/
void Inendi::PVView::set_color_on_post_filter_layer(const PVCore::PVHSVColor c)
{
	post_filter_layer.get_lines_properties().selection_set_color(post_filter_layer.get_selection(),
	                                                             get_row_count(), c);
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
int Inendi::PVView::set_layer_stack_layer_n_name(int n, QString const& name)
{
	PVLayer& layer = layer_stack.get_layer_n(n);
	layer.set_name(name);
	return 0;
}

/******************************************************************************
 *
 * Inendi::PVView::set_layer_stack_selected_layer_index
 *
 *****************************************************************************/
void Inendi::PVView::set_layer_stack_selected_layer_index(int index)
{
	layer_stack.set_selected_layer_index(index);
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
 * Inendi::PVView::toggle_layer_stack_layer_n_locked_state
 *
 *****************************************************************************/
int Inendi::PVView::toggle_layer_stack_layer_n_locked_state(int n)
{
	PVLayer& layer = layer_stack.get_layer_n(n);

	if (layer.get_locked()) {
		layer.set_locked(0);
	} else {
		layer.set_locked(1);
	}

	layer_stack.update_layer_index_array_completely();
	return 0;
}

/******************************************************************************
 *
 * Inendi::PVView::toggle_layer_stack_layer_n_visible_state
 *
 *****************************************************************************/
int Inendi::PVView::toggle_layer_stack_layer_n_visible_state(int n)
{
	PVLayer& layer = layer_stack.get_layer_n(n);

	if (layer.get_visible()) {
		layer.set_visible(0);
	} else {
		layer.set_visible(1);
	}

	layer_stack.update_layer_index_array_completely();
	return 0;
}

/******************************************************************************
 * Inendi::PVView::move_selected_layer_to
 *****************************************************************************/

void Inendi::PVView::move_selected_layer_to(int new_index)
{
	get_layer_stack().move_selected_layer_to(new_index);
}

void Inendi::PVView::set_consistent(bool c)
{
	_is_consistent = c;
}

bool Inendi::PVView::is_consistent() const
{
	// return get_plotted_parent()->is_uptodate() && _is_consistent;
	return _is_consistent;
}

void Inendi::PVView::recreate_mapping_plotting()
{
	// Source has been changed, recreate mapping and plotting
	get_parent<PVMapped>()->compute();
	get_parent<PVPlotted>()->process_from_parent_mapped();
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
	volatile_selection = ~floating_selection;
}

QString Inendi::PVView::get_name() const
{
	return QString("%1 (%2/%3)")
	    .arg(QString::number(get_display_view_id()))
	    .arg(get_parent<PVMapped>()->get_name())
	    .arg(get_parent<PVPlotted>()->get_name());
}

QString Inendi::PVView::get_window_name() const
{
	QString ret = get_parent<PVSource>()->get_window_name() + " | ";
	ret += get_name();
	return ret;
}

void Inendi::PVView::add_column(PVAxis const& axis)
{
	_axes_combination.axis_append(axis);
}

Inendi::PVSelection const* Inendi::PVView::get_selection_visible_listing() const
{
	if (_state_machine.are_listing_no_nu_nz()) {
		return &real_output_selection;
	}

	if (_state_machine.are_listing_no_nu()) {
		return &nu_selection;
	}

	if (_state_machine.are_listing_no_nz()) {
		return &layer_stack_output_layer.get_selection();
	}

	throw std::runtime_error("Invalid machine state");
}

void Inendi::PVView::toggle_listing_unselected_visibility()
{
	_state_machine.toggle_listing_unselected_visibility();
}

void Inendi::PVView::toggle_listing_zombie_visibility()
{
	_state_machine.toggle_listing_zombie_visibility();
}

void Inendi::PVView::toggle_view_unselected_zombie_visibility()
{
	_state_machine.toggle_view_unselected_zombie_visibility();
}

bool& Inendi::PVView::are_view_unselected_zombie_visible()
{
	return _state_machine.are_view_unselected_zombie_visible();
}

void Inendi::PVView::compute_layer_min_max(Inendi::PVLayer& layer)
{
	layer.compute_min_max(*get_parent<Inendi::PVPlotted>());
}

void Inendi::PVView::compute_selectable_count(Inendi::PVLayer& layer)
{
	layer.compute_selectable_count(get_row_count());
}

void Inendi::PVView::recompute_all_selectable_count()
{
	layer_stack.compute_selectable_count(get_row_count());
}

void Inendi::PVView::finish_process_from_rush_pipeline()
{
	layer_stack.compute_selectable_count(get_row_count());
}

void Inendi::PVView::set_axes_combination_list_id(PVAxesCombination::columns_indexes_t const& idxes,
                                                  PVAxesCombination::list_axes_t const& axes)
{
	_axes_combination.set_axes_index_list(idxes, axes);
}

PVRow Inendi::PVView::get_plotted_col_min_row(PVCol const combined_col) const
{
	PVCol const col = _axes_combination.get_axis_column_index(combined_col);
	return get_parent<PVPlotted>()->get_col_min_row(col);
}

PVRow Inendi::PVView::get_plotted_col_max_row(PVCol const combined_col) const
{
	PVCol const col = _axes_combination.get_axis_column_index(combined_col);
	return get_parent<PVPlotted>()->get_col_max_row(col);
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
	set_last_so(so.shared_from_this());
}

void Inendi::PVView::serialize_read(PVCore::PVSerializeObject& so,
                                    PVCore::PVSerializeArchive::version_t /*v*/)
{
	if (!so.object("layer-stack", layer_stack, "Layers", true)) {
		// If no layer stack, reset all layers so that we have one :)
		reset_layers();
	}
	set_row_count(layer_stack.get_layer_n(0).get_selection().count()); // please kill me
	so.object("axes-combination", _axes_combination, "Axes combination", true);
}
