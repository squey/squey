/**
 * \file PVView.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <math.h>

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVSerializeArchiveOptions.h>
#include <picviz/PVPlotted.h>
#include <picviz/PVRoot.h>
#include <picviz/PVSource.h>
#include <picviz/PVView.h>
#include <picviz/PVLayerFilter.h>
#include <picviz/PVMapped.h>
#include <picviz/PVMapping.h>
#include <picviz/PVPlotted.h>
#include <picviz/PVPlotting.h>

#include <tbb/tick_count.h>

/******************************************************************************
 *
 * Picviz::PVView::PVView
 *
 *****************************************************************************/
Picviz::PVView::PVView():
	pre_filter_layer("pre_filter_layer"),
	post_filter_layer("post_filter_layer"),
	layer_stack_output_layer("view_layer_stack_output_layer"),
	output_layer("output_layer"),
	_view_id(-1)
{
	init_defaults();
}

/*
Picviz::PVView::PVView(PVPlotted* parent) :
	pre_filter_layer("pre_filter_layer"),
	post_filter_layer("post_filter_layer"),
	layer_stack_output_layer("view_layer_stack_output_layer"),
	output_layer("output_layer"),
	_view_id(-1)
{
	set_parent(parent);

	init_defaults();
	init_from_plotted(parent, false);
}*/

Picviz::PVView::PVView(const PVView& /*org*/):
	pre_filter_layer("pre_filter_layer"),
	post_filter_layer("post_filter_layer"),
	layer_stack_output_layer("view_layer_stack_output_layer"),
	output_layer("output_layer"),
	_view_id(-1)
{
	assert(false);
}

void Picviz::PVView::set_parent_from_ptr(PVPlotted* plotted)
{
	data_tree_view_t::set_parent_from_ptr(plotted);

	_rushnraw_parent = &get_parent<PVSource>()->get_rushnraw();

	// Create layer filter arguments for that view
	LIB_CLASS(Picviz::PVLayerFilter) &filters_layer = 	LIB_CLASS(Picviz::PVLayerFilter)::get();
	LIB_CLASS(Picviz::PVLayerFilter)::list_classes const& lf = filters_layer.get_list();
	
	LIB_CLASS(Picviz::PVLayerFilter)::list_classes::const_iterator it;

	for (it = lf.begin(); it != lf.end(); it++) {
		filters_args[it.key()] = it.value()->get_default_args_for_view(*this);
	}
	row_count = get_parent<PVPlotted>()->get_row_count();
	layer_stack.set_row_count(row_count);
	eventline.set_row_count(row_count);
	eventline.set_first_index(0);
	eventline.set_current_index(row_count);
	eventline.set_last_index(row_count);

	reset_view();
}

void Picviz::PVView::process_parent_plotted()
{
	// Init default axes combination from source
	Picviz::PVPlotted* plotted = get_parent();
	PVSource* source = plotted->get_parent<PVSource>();
	axes_combination.set_from_format(source->get_format());
	axes_combination.set_axis_name(0, axes_combination.get_axis(0).get_name()); // Hack to detach QVector

	row_count = get_parent<PVPlotted>()->get_row_count();
	layer_stack.set_row_count(row_count);
	eventline.set_row_count(row_count);
	eventline.set_first_index(0);
	eventline.set_current_index(row_count);
	eventline.set_last_index(row_count);

	// First process
	//select_all_nonzb_lines(); Fixes bug #279
	nu_selection.select_none();

	process_from_layer_stack();

	_is_consistent = true;
}

void Picviz::PVView::reset_view()
{
	reset_layers();

	PVSource* source = get_parent<PVSource>();
	axes_combination = source->get_axes_combination();
	axes_combination.set_axis_name(0, axes_combination.get_axis(0).get_name()); // Hack to detach QVector
}

void Picviz::PVView::set_fake_axes_comb(PVCol const ncols)
{
	axes_combination.clear();
	for (PVCol c = 0; c < ncols; c++) {
		PVAxis axis;
		axis.set_name(QString("axis ") + QString::number(c));
		axis.set_titlecolor("#ffffff");
		axes_combination.axis_append(axis);
	}
}

/******************************************************************************
 *
 * Picviz::PVView::~PVView
 *
 *****************************************************************************/
Picviz::PVView::~PVView()
{
	PVLOG_DEBUG("In PVView destructor: 0x%x\n", this);
	PVRoot* root = get_parent<PVRoot>();
	if (root) {
		root->remove_view_from_correlations(this);
		root->view_being_deleted(this);
	}
	delete state_machine;
}

/******************************************************************************
 *
 * Picviz::PVView::init_defaults
 *
 *****************************************************************************/
void Picviz::PVView::init_defaults()
{
	_is_consistent = false;
	_active_axis = 0;
	_rushnraw_parent = NULL;

	last_extractor_batch_size = pvconfig.value("pvkernel/rush/extract_next", PVEXTRACT_NUMBER_LINES_NEXT_DEFAULT).toInt();

	state_machine = new Picviz::PVStateMachine();

	default_zombie_line_properties.h() = HSV_COLOR_BLACK;
}

/******************************************************************************
 *
 * Picviz::PVView::reset_layers
 *
 *****************************************************************************/
void Picviz::PVView::reset_layers()
{
	// This function remove all the layers and add the default one with all events selected
	bool old_consistent = _is_consistent;
	_is_consistent = false;
	layer_stack.delete_all_layers();
	layer_stack.append_new_layer();
	layer_stack.get_layer_n(0).reset_to_full_and_default_color();
	if (row_count != 0) {
		/* when a .pvi is loaded, the mapped and the plotted are
		 * uninitialized when the view is created (the rush pipeline
		 * is runned later).
		 */
		layer_stack.get_layer_n(0).compute_selectable_count(row_count);
	}
	pre_filter_layer.reset_to_full_and_default_color();
	post_filter_layer.reset_to_full_and_default_color();
	layer_stack_output_layer.reset_to_full_and_default_color();
	output_layer.reset_to_full_and_default_color();

	_is_consistent = old_consistent;
}

/******************************************************************************
 *
 * Picviz::PVView::add_new_layer
 *
 *****************************************************************************/
void Picviz::PVView::add_new_layer(QString name)
{
	Picviz::PVLayer* layer = layer_stack.append_new_layer(name);
	layer->compute_selectable_count(row_count);
}

void Picviz::PVView::add_new_layer_from_file(const QString& path)
{
	// Create a new layer
	Picviz::PVLayer* layer = layer_stack.append_new_layer();

	// And load it
	layer->load_from_file(path);
	layer->compute_min_max(*get_parent<Picviz::PVPlotted>());
	layer->compute_selectable_count(get_parent<Picviz::PVPlotted>()->get_row_count());
}

void Picviz::PVView::delete_selected_layer()
{
	layer_stack.delete_selected_layer();
}

void Picviz::PVView::delete_layer_n(int idx)
{
	layer_stack.delete_by_index(idx);
}

void Picviz::PVView::duplicate_selected_layer(const QString &name)
{
	PVLayer* new_layer = layer_stack.duplicate_selected_layer(name);
	compute_layer_min_max(*new_layer);
	new_layer->compute_selectable_count(row_count);
}

void Picviz::PVView::load_from_file(const QString& file)
{
	layer_stack.load_from_file(file);
	layer_stack.compute_min_maxs(*get_parent<Picviz::PVPlotted>());
	layer_stack.compute_selectable_count(get_parent<Picviz::PVPlotted>()->get_row_count());
}

/******************************************************************************
 *
 * Picviz::PVView::apply_filter_named_select_all
 *
 *****************************************************************************/
void Picviz::PVView::apply_filter_named_select_all()
{
	pre_filter_layer = layer_stack_output_layer;
	post_filter_layer.get_selection().select_all();
}

/******************************************************************************
 *
 * Picviz::PVView::commit_to_new_layer
 *
 *****************************************************************************/
void Picviz::PVView::commit_to_new_layer()
{
	const PVSelection& sel = post_filter_layer.get_selection();
	const PVLinesProperties &lp = output_layer.get_lines_properties();

	PVLayer *layer = layer_stack.append_new_layer_from_selection_and_lines_properties(sel, lp);
	layer->compute_min_max(*get_parent<Picviz::PVPlotted>());
	layer->compute_selectable_count(row_count);
}

void Picviz::PVView::commit_volatile_in_floating_selection()
{
	switch (state_machine->get_square_area_mode()) {
		case Picviz::PVStateMachine::AREA_MODE_ADD_VOLATILE:
			floating_selection |= volatile_selection;
			break;

		case Picviz::PVStateMachine::AREA_MODE_INTERSECT_VOLATILE:
			floating_selection &= volatile_selection;
			break;

		case Picviz::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE:
			floating_selection = volatile_selection;
			break;

		case Picviz::PVStateMachine::AREA_MODE_SUBSTRACT_VOLATILE:
			floating_selection -= volatile_selection;
			break;

		case Picviz::PVStateMachine::AREA_MODE_OFF:
			;
	}
}

/******************************************************************************
 *
 * Picviz::PVView::debug
 *
 *****************************************************************************/
void Picviz::PVView::debug()
{

}

/******************************************************************************
 *
 * Picviz::PVView::expand_selection_on_axis
 *
 *************************************************************************data_tree_view_t****/
void Picviz::PVView::expand_selection_on_axis(PVCol axis_id, QString const& mode)
{
	commit_volatile_in_floating_selection();
	get_parent<PVPlotted>()->expand_selection_on_axis(floating_selection, axis_id, mode);
}

/******************************************************************************
 *
 * Picviz::PVView::get_axes_count
 *
 *****************************************************************************/
PVCol Picviz::PVView::get_axes_count() const
{
	return axes_combination.get_axes_count();
}

/******************************************************************************
 *
 * Picviz::PVView::get_axes_names_list
 *
 *****************************************************************************/
QStringList Picviz::PVView::get_axes_names_list() const
{
	return axes_combination.get_axes_names_list();
}

QStringList Picviz::PVView::get_zones_names_list() const
{
	const QStringList axes = get_axes_names_list();

	QStringList ret;
	ret.reserve(axes.size() - 1);

	const QString del(" <-> ");

	for (int i = 0; i < axes.size()-1; i++) {
		ret << axes[i] + del + axes[i+1];
	}

	return ret;
}

Picviz::PVAxis const& Picviz::PVView::get_axis(PVCol index) const
{
	return axes_combination.get_axis(index);
}

Picviz::PVAxis const& Picviz::PVView::get_axis_by_id(axes_comb_id_t const axes_comb_id) const
{
	return axes_combination.get_axis(axes_combination.get_index_by_id(axes_comb_id));
}

/******************************************************************************
 *
 * Picviz::PVView::get_axis_name
 *
 *****************************************************************************/
const QString& Picviz::PVView::get_axis_name(PVCol index) const
{
	PVAxis const& axis = axes_combination.get_axis(index);
	return axis.get_name();
}

QString Picviz::PVView::get_axis_type(PVCol index) const
{
	PVAxis const& axis = axes_combination.get_axis(index);
	return axis.get_type();
}

QString Picviz::PVView::get_original_axis_name(PVCol axis_id) const
{
	PVAxis const& axis = axes_combination.get_original_axis(axis_id);
	return axis.get_name();
}

QString Picviz::PVView::get_original_axis_type(PVCol axis_id) const
{
	PVAxis const& axis = axes_combination.get_original_axis(axis_id);
	return axis.get_type();
}

// FIXME: This function should be removed
/******************************************************************************
 *
 * Picviz::PVView::get_color_in_output_layer
 *
 *****************************************************************************/
const PVCore::PVHSVColor Picviz::PVView::get_color_in_output_layer(PVRow index) const
{
	return output_layer.get_lines_properties().get_line_properties(index);
}

/******************************************************************************
 *
 * Picviz::PVView::get_column_count
 *
 *****************************************************************************/
PVCol Picviz::PVView::get_column_count() const
{
	return get_axes_combination().get_axes_count();
}

/******************************************************************************
 *
 * Picviz::PVView::get_column_count_as_float
 *
 *****************************************************************************/
float Picviz::PVView::get_column_count_as_float()
{
	// AG: why?
	return (float)get_column_count();
}


/******************************************************************************
 *
 * Picviz::PVView::get_data
 *
 *****************************************************************************/
QString Picviz::PVView::get_data(PVRow row, PVCol column) const
{
	PVCol real_index = axes_combination.get_axis_column_index_fast(column);

	return get_rushnraw_parent().at(row, real_index);
}

PVCore::PVUnicodeString Picviz::PVView::get_data_unistr(PVRow row, PVCol column) const
{
	PVCol real_index = axes_combination.get_axis_column_index_fast(column);
	return get_rushnraw_parent().at_unistr(row, real_index);
}

/******************************************************************************
 *
 * Picviz::PVView::get_real_axis_index
 *
 *****************************************************************************/
PVCol Picviz::PVView::get_real_axis_index(PVCol col) const
{
	return axes_combination.get_axis_column_index(col);
}

/******************************************************************************
 *
 * Picviz::PVView::get_layer_index
 *
 *****************************************************************************/
int Picviz::PVView::get_layer_index(int index)
{
	return layer_stack.get_lia().get_value(index);
}

/******************************************************************************
 *
 * Picviz::PVView::get_layer_index_as_float
 *
 *****************************************************************************/
float Picviz::PVView::get_layer_index_as_float(int index)
{
	return (float)get_layer_index(index);
}

/******************************************************************************
 *
 * Picviz::PVView::get_layer_stack
 *
 *****************************************************************************/
Picviz::PVLayerStack &Picviz::PVView::get_layer_stack()
{
	return layer_stack;
}

/******************************************************************************
 *
 * Picviz::PVView::get_layer_stack_layer_n_locked_state
 *
 *****************************************************************************/
int Picviz::PVView::get_layer_stack_layer_n_locked_state(int n) const
{
	PVLayer const& layer = layer_stack.get_layer_n(n);
	return layer.get_locked();
}

/******************************************************************************
 *
 * Picviz::PVView::get_layer_stack_layer_n_name
 *
 *****************************************************************************/
QString Picviz::PVView::get_layer_stack_layer_n_name(int n) const
{
	PVLayer const& layer = layer_stack.get_layer_n(n);
	return layer.get_name();
}

/******************************************************************************
 *
 * Picviz::PVView::get_layer_stack_layer_n_visible_state
 *
 *****************************************************************************/
int Picviz::PVView::get_layer_stack_layer_n_visible_state(int n) const
{
	PVLayer const& layer = layer_stack.get_layer_n(n);
	return layer.get_visible();
}

/******************************************************************************
 *
 * Picviz::PVView::get_layer_stack_output_layer
 *
 *****************************************************************************/
Picviz::PVLayer &Picviz::PVView::get_layer_stack_output_layer()
{
	return layer_stack_output_layer;
}

/******************************************************************************
 *
 * Picviz::PVView::get_line_state_in_layer_stack_output_layer
 *
 *****************************************************************************/
bool Picviz::PVView::get_line_state_in_layer_stack_output_layer(PVRow index)
{
	return layer_stack_output_layer.get_selection().get_line(index);
}

bool Picviz::PVView::get_line_state_in_layer_stack_output_layer(PVRow index) const
{
	return layer_stack_output_layer.get_selection().get_line(index);
}

/******************************************************************************
 *
 * Picviz::PVView::get_line_state_in_output_layer
 *
 *****************************************************************************/
bool Picviz::PVView::get_line_state_in_output_layer(PVRow index)
{
	return output_layer.get_selection().get_line(index);
}

bool Picviz::PVView::get_line_state_in_output_layer(PVRow index) const
{
	return output_layer.get_selection().get_line(index);
}

/******************************************************************************
 *
 * Picviz::PVView::get_line_state_in_pre_filter_layer
 *
 *****************************************************************************/
bool Picviz::PVView::get_line_state_in_pre_filter_layer(PVRow index)
{
	return pre_filter_layer.get_selection().get_line(index);
}

bool Picviz::PVView::get_line_state_in_pre_filter_layer(PVRow index) const
{
	return pre_filter_layer.get_selection().get_line(index);
}

/******************************************************************************
 *
 * Picviz::PVView::get_nu_selection
 *
 *****************************************************************************/
Picviz::PVSelection &Picviz::PVView::get_nu_selection()
{
	return nu_selection;
}

/******************************************************************************
 *
 * Picviz::PVView::get_number_of_selected_lines
 *
 *****************************************************************************/
int Picviz::PVView::get_number_of_selected_lines()
{
	return real_output_selection.get_number_of_selected_lines_in_range(0, row_count);
}

/******************************************************************************
 *
 * Picviz::PVView::get_original_axes_count
 *
 *****************************************************************************/
PVCol Picviz::PVView::get_original_axes_count() const
{
	return axes_combination.get_original_axes_count();
}

/******************************************************************************
 *
 * Picviz::PVView::get_output_layer
 *
 *****************************************************************************/
Picviz::PVLayer &Picviz::PVView::get_output_layer()
{
	return output_layer;
}

/******************************************************************************
 *
 * Picviz::PVView::get_post_filter_layer
 *
 *****************************************************************************/
Picviz::PVLayer &Picviz::PVView::get_post_filter_layer()
{
	return post_filter_layer;
}

/******************************************************************************
 *
 * Picviz::Picviz::PVView::get_pre_filter_layer
 *
 *****************************************************************************/
Picviz::PVLayer &Picviz::PVView::get_pre_filter_layer()
{
	return pre_filter_layer;
}

/******************************************************************************
 *
 * Picviz::PVView::get_real_output_selection
 *
 *****************************************************************************/
Picviz::PVSelection &Picviz::PVView::get_real_output_selection()
{
	return real_output_selection;
}

Picviz::PVSelection const& Picviz::PVView::get_real_output_selection() const
{
	return real_output_selection;
}

/******************************************************************************
 *
 * Picviz::PVView::get_row_count
 *
 *****************************************************************************/
PVRow Picviz::PVView::get_row_count() const
{
	return get_parent<PVPlotted>()->get_row_count();
}

/******************************************************************************
 *
 * Picviz::PVView::load_post_to_pre
 *
 *****************************************************************************/
void Picviz::PVView::load_post_to_pre()
{
	pre_filter_layer = post_filter_layer;
}

/******************************************************************************
 *
 * Picviz::PVView::move_active_axis_closest_to_position
 *
 *****************************************************************************/
int Picviz::PVView::move_active_axis_closest_to_position(float x)
{
	/* CODE */
	PVCol new_index = get_active_axis_closest_to_position(x);

	/* We move the axis if there is a movement */
	if ( new_index != _active_axis ) {
		axes_combination.move_axis_to_new_position(_active_axis, new_index);
		_active_axis = new_index;

		return 1;
	} else {
		return 0;
	}
}

/******************************************************************************
 *
 * Picviz::PVView::get_active_axis_closest_to_position
 *
 *****************************************************************************/
PVCol Picviz::PVView::get_active_axis_closest_to_position(float x)
{
	PVCol axes_count = axes_combination.get_axes_count();
	int ret = (int)floor(x + 0.5);
	if (ret < 0) {
		/* We set the leftmost AXIS as destination */
		return 0;
	}
	else
	if ( ret >= axes_count ) {
		/* We set the rightmost AXIS as destination */
		return (PVCol)(axes_count - 1);
	}

	return (PVCol) ret;
}

/******************************************************************************
 *
 * Picviz::PVView::process_eventline
 *
 *****************************************************************************/
void Picviz::PVView::process_eventline()
{
	/* We compute the real_output_selection */
	eventline.selection_A2B_filter(post_filter_layer.get_selection(), real_output_selection);

	/* We refresh the nu_selection */
	nu_selection = std::move(~layer_stack_output_layer.get_selection());
	
	nu_selection |= real_output_selection;

	PVLinesProperties& out_lps = output_layer.get_lines_properties();
	PVLinesProperties const& post_lps = post_filter_layer.get_lines_properties();
	/* We are now able to process the lines_properties */
	for (PVRow i = 0; i < row_count; i++) {
		/* We check if the event is selected at the end of the process */
		PVCore::PVHSVColor& out_lp = out_lps.get_line_properties(i);
		PVCore::PVHSVColor const& post_lp = post_lps.get_line_properties(i);
		if (real_output_selection.get_line(i)) {
			/* It is selected, so we copy its line properties */
			out_lp = post_lp;
		} else {
			/* It is not selected in the end, so we check if it was available in the beginning */
			if (layer_stack_output_layer.get_selection().get_line(i)) {
				/* The event was available, but is unselected */
				out_lp = post_lp;
			} else {
				/* The event is a zombie one */
				out_lp = default_zombie_line_properties;
			}
		}
	}
}

/******************************************************************************
 *
 * Picviz::PVView::process_filter
 *
 *****************************************************************************/
void Picviz::PVView::process_filter()
{
	// FIXME: This is temporary!!! -> AG: why ?
	post_filter_layer = pre_filter_layer;
}

/******************************************************************************
 *
 * Picviz::PVView::process_correlation
 *
 *****************************************************************************/
QList<Picviz::PVView*> Picviz::PVView::process_correlation()
{
	Picviz::PVRoot* root = get_parent<Picviz::PVRoot>();
	// AG: in some test cases, there is no PVRoot!
	if (root) {
		return root->process_correlation(this);
	}
	return QList<Picviz::PVView*>();
}

/******************************************************************************
 *
 * Picviz::PVView::process_from_eventline
 *
 *****************************************************************************/
void Picviz::PVView::process_from_eventline()
{
	process_eventline();
	process_visibility();
}

/******************************************************************************
 *
 * Picviz::PVView::process_from_filter
 *
 *****************************************************************************/
void Picviz::PVView::process_from_filter()
{
	process_filter();
	process_eventline();
	process_visibility();
}

/******************************************************************************
 *
 * Picviz::PVView::process_from_layer_stack
 *
 *****************************************************************************/
QList<Picviz::PVView*> Picviz::PVView::process_from_layer_stack()
{
	tbb::tick_count start = tbb::tick_count::now();

	/* We start by reprocessing the layer_stack */
	process_layer_stack();
	process_selection();
	process_filter();
	process_eventline();
	process_visibility();
	QList<Picviz::PVView*> changed_views = process_correlation();

	tbb::tick_count end = tbb::tick_count::now();
	PVLOG_INFO("(Picviz::PVView::process_from_layer_stack) function took %0.4f seconds.\n", (end-start).seconds());

	return changed_views;
}

/******************************************************************************
 *
 * Picviz::PVView::process_from_selection
 *
 *****************************************************************************/
QList<Picviz::PVView*> Picviz::PVView::process_from_selection()
{
	PVLOG_DEBUG("Picviz::PVView::%s\n",__FUNCTION__);
	process_selection();
	process_filter();
	process_eventline();
	process_visibility();
	QList<Picviz::PVView*> changed_views = process_correlation();

	return changed_views;
}

QList<Picviz::PVView*> Picviz::PVView::process_real_output_selection()
{
	// AG: TODO: should be optimised to only create real_output_selection
	QList<Picviz::PVView*> changed_views = process_from_selection();
	return changed_views;
}

/******************************************************************************
 *
 * Picviz::PVView::process_layer_stack
 *
 *****************************************************************************/
void Picviz::PVView::process_layer_stack()
{
	layer_stack.process(layer_stack_output_layer, get_row_count());
}

/******************************************************************************
 *
 * Picviz::PVView::process_selection
 *
 *****************************************************************************/
void Picviz::PVView::process_selection()
{
	/* We treat the selection according to specific SQUARE_AREA_SELECTION mode */
	switch (state_machine->get_square_area_mode()) {
		case Picviz::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE:
			pre_filter_layer.get_selection() = volatile_selection;
//			volatile_selection.A2B_copy(pre_filter_layer->selection);
			break;

		case Picviz::PVStateMachine::AREA_MODE_ADD_VOLATILE:
			pre_filter_layer.get_selection() = std::move(floating_selection | volatile_selection);
//			floating_selection.AB2C_or(volatile_selection, pre_filter_layer->selection);
			break;

		case Picviz::PVStateMachine::AREA_MODE_SUBSTRACT_VOLATILE:
			// This is an optimized version of:
			// pre_filter_layer.get_selection() = floating_selection, volatile_selection
			pre_filter_layer.get_selection().AB_sub(floating_selection, volatile_selection);
			break;

		case Picviz::PVStateMachine::AREA_MODE_INTERSECT_VOLATILE:
			pre_filter_layer.get_selection() = std::move(floating_selection & volatile_selection);
		//	floating_selection.AB2C_and(volatile_selection, pre_filter_layer->selection);
			break;

		default:
			pre_filter_layer.get_selection() = layer_stack_output_layer.get_selection();
		//	layer_stack_output_layer->selection.A2B_copy(pre_filter_layer->selection);
			break;
	}

	/* We cut the resulting selection with what is available in the layer_stack */
	/* We test if we are in ALL edit_mode or SOLO edit_mode */
	if (state_machine->is_edit_mode_all()) {
		//PVLOG_INFO("(process_selection) we are in edit mode all. Remove me !\n");
		/* We are in ALL edit_mode */
		pre_filter_layer.get_selection() &= layer_stack_output_layer.get_selection();
	} else {
		/* We are in SOLO edit_mode */
		pre_filter_layer.get_selection() &= layer_stack.get_selected_layer().get_selection();
	}

	/* We simply copy the lines_properties */
	// picviz_lines_properties_A2B_copy(layer_stack_output_layer->lines_properties, pre_filter_layer->lines_properties);
	//std::swap(pre_filter_layer.get_lines_properties(), layer_stack_output_layer.get_lines_properties());
	pre_filter_layer.get_lines_properties() = layer_stack_output_layer.get_lines_properties();

	/* Now we MUST refresh the index_array associated to nznu */
	/* WARNING ! nothing is done here because this function should be followed by process_eventline which changes the selection and DO update the nznu_index_array */
}

/******************************************************************************
 *
 * Picviz::PVView::process_visibility
 *
 *****************************************************************************/
void Picviz::PVView::process_visibility()
{
    PVLOG_DEBUG("Picviz::PVView::process_visibility\n");
	/* We First need to check if the UNSELECTED lines are visible */
	if (state_machine->are_listing_unselected_visible()) {
		/* The UNSELECTED are visible */
		/* Now we need to know if the ZOMBIE are visible */
		if (state_machine->are_listing_zombie_visible()) {
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
		if (state_machine->are_listing_zombie_visible()) {
			/* ZOMBIE are visible */
			output_layer.get_selection().or_not(layer_stack_output_layer.get_selection());
		}
	}
}

/******************************************************************************
 *
 * Picviz::PVView::set_active_axis_closest_to_position
 *
 *****************************************************************************/
void Picviz::PVView::set_active_axis_closest_to_position(float x)
{
	/* VARIABLES */
	int closest_int;
	int axes_count;
	
	/* CODE */
	closest_int = (int)floor(x + 0.5);
	if ( closest_int < 0 ) {
		/* We set the leftmost AXIS as active */
		_active_axis = 0;
	} else {
		axes_count = axes_combination.get_axes_count();
		if ( closest_int >= axes_count ) {
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
 * Picviz::PVView::set_axis_name
 *
 *****************************************************************************/
void Picviz::PVView::set_axis_name(PVCol index, const QString &name_)
{
	PVAxis axis;

	axes_combination.set_axis_name(index, name_);
}

/******************************************************************************
 *
 * Picviz::PVView::set_color_on_active_layer
 *
 *****************************************************************************/
void Picviz::PVView::set_color_on_active_layer(const PVCore::PVHSVColor c)
{
	/* VARIABLES */
	PVLayer &active_layer = layer_stack.get_selected_layer();

	active_layer.get_lines_properties().selection_set_color(get_real_output_selection(), row_count, c);
}

/******************************************************************************
 *
 * Picviz::PVView::set_color_on_post_filter_layer
 *
 *****************************************************************************/
void Picviz::PVView::set_color_on_post_filter_layer(const PVCore::PVHSVColor c)
{
	post_filter_layer.get_lines_properties().selection_set_color(post_filter_layer.get_selection(), row_count, c);
}

/******************************************************************************
 *
 * Picviz::PVView::set_floating_selection
 *
 *****************************************************************************/
void Picviz::PVView::set_floating_selection(PVSelection &selection)
{
	floating_selection = selection;
//	selection.A2B_copy(floating_selection);
}

/******************************************************************************
 *
 * Picviz::PVView::set_layer_stack_layer_n_name
 *
 *****************************************************************************/
int Picviz::PVView::set_layer_stack_layer_n_name(int n, QString const& name)
{
	PVLayer &layer = layer_stack.get_layer_n(n);
	layer.set_name(name);
	return 0;
}

/******************************************************************************
 *
 * Picviz::PVView::set_layer_stack_selected_layer_index
 *
 *****************************************************************************/
void Picviz::PVView::set_layer_stack_selected_layer_index(int index)
{
	layer_stack.set_selected_layer_index(index);
}

/******************************************************************************
 *
 * Picviz::PVView::set_selection_with_final_selection
 *
 *****************************************************************************/
void Picviz::PVView::set_selection_with_final_selection(PVSelection &selection)
{
	selection = post_filter_layer.get_selection();
//	post_filter_layer->selection.A2B_copy(selection);
	eventline.selection_A2A_filter(selection);
}

/******************************************************************************
 *
 * Picviz::PVView::set_selection_from_layer
 *
 *****************************************************************************/
void Picviz::PVView::set_selection_from_layer(PVLayer const& layer)
{
	set_selection_view(layer.get_selection());
}

/******************************************************************************
 *
 * Picviz::PVView::set_selection_view
 *
 *****************************************************************************/
void Picviz::PVView::set_selection_view(PVSelection const& sel)
{
	state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE);
	volatile_selection = sel;
}
/******************************************************************************
 *
 * Picviz::PVView::sortByColumn
 *
 *****************************************************************************/
/*void Picviz::PVView::sortByColumn(int idColumn){
	PVLOG_INFO("Picviz::PVView::sortByColumn(%d)\n",idColumn);
	//init
	QVector<QStringList> nraw;
	nraw = get_qtnraw_parent();
	//sorting
	PVSortQVectorQStringList sorter;
	sorter.setList(&nraw);
	sorter.sortByColumn(idColumn);
}*/

/******************************************************************************
 *
 * Picviz::PVView::toggle_layer_stack_layer_n_locked_state
 *
 *****************************************************************************/
int Picviz::PVView::toggle_layer_stack_layer_n_locked_state(int n)
{
	PVLayer &layer = layer_stack.get_layer_n(n);

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
 * Picviz::PVView::toggle_layer_stack_layer_n_visible_state
 *
 *****************************************************************************/
int Picviz::PVView::toggle_layer_stack_layer_n_visible_state(int n)
{
	PVLayer &layer = layer_stack.get_layer_n(n);
	
	if (layer.get_visible()) {
		layer.set_visible(0);
	} else {
		layer.set_visible(1);
	}
	
	layer_stack.update_layer_index_array_completely();
	return 0;
}

PVRush::PVExtractor& Picviz::PVView::get_extractor()
{
	return get_parent<PVSource>()->get_extractor();
}

void Picviz::PVView::set_consistent(bool c)
{
	_is_consistent = c;
}

bool Picviz::PVView::is_consistent() const
{
	//return get_plotted_parent()->is_uptodate() && _is_consistent;
	return _is_consistent;
}

void Picviz::PVView::recreate_mapping_plotting()
{
	// Source has been changed, recreate mapping and plotting
	get_parent<PVMapped>()->process_parent_source();
	get_parent<PVPlotted>()->process_from_parent_mapped();

/*
	// Save current axes combination
	PVAxesCombination cur_axes_combination = axes_combination;

	// Reiinit the view with the new plotted
	init_from_plotted(plotted, false);

	// Restore the previous axes combination
	axes_combination = cur_axes_combination;
*/
}

void Picviz::PVView::select_all_nonzb_lines()
{
	state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE);
	volatile_selection.select_all();
}

void Picviz::PVView::select_no_line()
{
	// Set square area mode w/ volatile
	state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE);
	volatile_selection.select_none();
}

void Picviz::PVView::select_inv_lines()
{
	// Commit current volatile selection
	commit_volatile_in_floating_selection();
	// Set square area mode w/ volatile
	state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE);
	volatile_selection = ~floating_selection;
}

QString Picviz::PVView::get_name() const
{
	return QString("%1 (%2/%3)").arg(QString::number(get_display_view_id())).arg(get_parent<PVMapped>()->get_name()).arg(get_parent<PVPlotted>()->get_name());
}

QString Picviz::PVView::get_window_name() const
{
	QString ret = get_parent<PVSource>()->get_window_name() + " | ";
	ret += get_name();
	return ret;
}

void Picviz::PVView::add_column(PVAxis const& axis)
{
	axes_combination.axis_append(axis);
}

Picviz::PVSelection const* Picviz::PVView::get_selection_visible_listing() const
{
	if (state_machine->are_listing_no_nu_nz()) {
		return &real_output_selection;
	}

	if (state_machine->are_listing_no_nu()) {
		return &nu_selection;
	}

	if (state_machine->are_listing_no_nz()) {
		return &layer_stack_output_layer.get_selection();
	}

	// If we're here, then we are listing all...
	return NULL;
}

bool Picviz::PVView::is_line_visible_listing(PVRow index) const
{
	if (state_machine->are_listing_all()) {
		return true;
	}

	if (state_machine->are_listing_no_nu_nz()) {
		return real_output_selection.get_line(index);
	}

	if (state_machine->are_listing_no_nu()) {
		return nu_selection.get_line(index);
	}

	if (state_machine->are_listing_no_nz()) {
		return layer_stack_output_layer.get_selection().get_line(index);
	}

	//return real_output_selection.get_line(index);
	return true;
}

bool Picviz::PVView::is_real_output_selection_empty() const
{
	return real_output_selection.is_empty();
}

void Picviz::PVView::toggle_listing_unselected_visibility()
{
	state_machine->toggle_listing_unselected_visibility();
}

void Picviz::PVView::toggle_listing_zombie_visibility()
{
	state_machine->toggle_listing_zombie_visibility();
}

void Picviz::PVView::toggle_view_unselected_zombie_visibility()
{
	state_machine->toggle_view_unselected_zombie_visibility();
}

bool& Picviz::PVView::are_view_unselected_zombie_visible()
{
	return state_machine->are_view_unselected_zombie_visible();
}

Picviz::PVSortingFunc_p Picviz::PVView::get_sort_plugin_for_col(PVCol col) const
{
	// Temporary, waiting for all of this to be configurable
	QString type = get_original_axis_type(col);
	PVSortingFunc_p f_lib = LIB_CLASS(Picviz::PVSortingFunc)::get().get_class_by_name(type + "_default");
	if (!f_lib) {
		f_lib = PVSortingFunc_p(new PVDefaultSortingFunc());
	}
	return f_lib;
}

void Picviz::PVView::compute_layer_min_max(Picviz::PVLayer& layer)
{
	layer.compute_min_max(*get_parent<Picviz::PVPlotted>());
}

void Picviz::PVView::compute_selectable_count(Picviz::PVLayer& layer)
{
	layer.compute_selectable_count(get_parent<Picviz::PVPlotted>()->get_row_count());
}

void Picviz::PVView::recompute_all_selectable_count()
{
	layer_stack.compute_selectable_count(get_row_count());
}

void Picviz::PVView::finish_process_from_rush_pipeline()
{
	layer_stack.compute_selectable_count(get_parent<PVPlotted>()->get_row_count());
}

void Picviz::PVView::set_axes_combination_list_id(PVAxesCombination::columns_indexes_t const& idxes, PVAxesCombination::list_axes_t const& axes)
{
	get_axes_combination().set_axes_index_list(idxes, axes);
}

PVRow Picviz::PVView::get_plotted_col_min_row(PVCol const combined_col) const
{
	PVCol const col = axes_combination.get_axis_column_index(combined_col);
	return get_parent<PVPlotted>()->get_col_min_row(col);
}

PVRow Picviz::PVView::get_plotted_col_max_row(PVCol const combined_col) const
{
	PVCol const col = axes_combination.get_axis_column_index(combined_col);
	return get_parent<PVPlotted>()->get_col_max_row(col);
}

// Load/save and serialization
void Picviz::PVView::serialize_write(PVCore::PVSerializeObject& so)
{
	so.object("layer-stack", layer_stack, "Layers", true);
	so.object("axes-combination", axes_combination, "Axes combination", true);
	set_last_so(so.shared_from_this());
}

void Picviz::PVView::serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	if (!so.object("layer-stack", layer_stack, "Layers", true)) {
		// If no layer stack, reset all layers so that we have one :)
		reset_layers();
	}
	so.object("axes-combination", axes_combination, "Axes combination", true);
}
