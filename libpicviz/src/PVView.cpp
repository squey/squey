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
#include <picviz/PVSortQVectorQStringList.h>

#include <tbb/tick_count.h>

/******************************************************************************
 *
 * Picviz::PVView::PVView
 *
 *****************************************************************************/
Picviz::PVView::PVView():
	boost::enable_shared_from_this<PVView>(),
	pre_filter_layer("pre_filter_layer"),
	post_filter_layer("post_filter_layer"),
	layer_stack_output_layer("view_layer_stack_output_layer"),
	output_layer("output_layer")
{
	init_defaults();
}

Picviz::PVView::PVView(PVPlotted* parent) :
	pre_filter_layer("pre_filter_layer"),
	post_filter_layer("post_filter_layer"),
	layer_stack_output_layer("view_layer_stack_output_layer"),
	output_layer("output_layer")
{
	init_defaults();
	init_from_plotted(parent, false);
}

Picviz::PVView::PVView(const PVView& /*org*/):
	boost::enable_shared_from_this<PVView>(),
	pre_filter_layer("pre_filter_layer"),
	post_filter_layer("post_filter_layer"),
	layer_stack_output_layer("view_layer_stack_output_layer"),
	output_layer("output_layer")
{
	assert(false);
}
/******************************************************************************
 *
 * Picviz::PVView::~PVView
 *
 *****************************************************************************/
Picviz::PVView::~PVView()
{
	PVLOG_INFO("In PVView destructor\n");
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
	active_axis = 0;

	last_extractor_batch_size = pvconfig.value("pvkernel/rush/extract_next", PVEXTRACT_NUMBER_LINES_NEXT_DEFAULT).toInt();

	state_machine = new Picviz::PVStateMachine();

	default_zombie_line_properties.r() = (unsigned char)0;
	default_zombie_line_properties.g() = (unsigned char)0;
	default_zombie_line_properties.b() = (unsigned char)0;
}

/******************************************************************************
 *
 * Picviz::PVView::init_from_plotted
 *
 *****************************************************************************/
void Picviz::PVView::init_from_plotted(PVPlotted* parent, bool keep_layers)
{
	root = parent->get_root_parent();
	plotted = parent;

	// Init default axes combination from source
	if (keep_layers) {
		axes_combination.set_from_format(parent->get_source_parent()->get_format());
	}
	else {
		axes_combination = parent->get_source_parent()->get_axes_combination();
	}

	// Create layer filter arguments for that view
	LIB_CLASS(Picviz::PVLayerFilter) &filters_layer = 	LIB_CLASS(Picviz::PVLayerFilter)::get();
	LIB_CLASS(Picviz::PVLayerFilter)::list_classes const& lf = filters_layer.get_list();
	
	LIB_CLASS(Picviz::PVLayerFilter)::list_classes::const_iterator it;

	for (it = lf.begin(); it != lf.end(); it++) {
		filters_args[it.key()] = it.value()->get_default_args_for_view(*this);
	}

	row_count = plotted->get_row_count();
	layer_stack.set_row_count(row_count);
	eventline.set_row_count(row_count);
	eventline.set_first_index(0);
	eventline.set_current_index(row_count);
	eventline.set_last_index(row_count);
	z_level_array.set_row_count(row_count);

	// First process
	if (!keep_layers) {
		reset_layers();
	}
	else {
		layer_stack.compute_min_maxs(*parent);
	}
	select_all_nonzb_lines();
	nu_selection.select_none();

	process_from_layer_stack();

	_is_consistent = true;
}

/******************************************************************************
 *
 * Picviz::PVView::reset_layers
 *
 *****************************************************************************/
void Picviz::PVView::reset_layers()
{
	// This function remove all the layers and add the default one with all lines selected
	bool old_consistent = _is_consistent;
	_is_consistent = false;
	layer_stack.delete_all_layers();
	layer_stack.append_new_layer();
	layer_stack.get_layer_n(0).get_selection().select_all();
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
int Picviz::PVView::add_new_layer()
{
	layer_stack.append_new_layer();

	return 0;
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

	layer_stack.append_new_layer_from_selection_and_lines_properties(sel, lp);
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
 *****************************************************************************/
void Picviz::PVView::expand_selection_on_axis(PVCol axis_id, QString const& mode)
{
	commit_volatile_in_floating_selection();
	get_plotted_parent()->expand_selection_on_axis(floating_selection, axis_id, mode);
}

/******************************************************************************
 *
 * Picviz::PVView::get_axes_count
 *
 *****************************************************************************/
PVCol Picviz::PVView::get_axes_count()
{
	return axes_combination.get_axes_count();
}

/******************************************************************************
 *
 * Picviz::PVView::get_axes_names_list
 *
 *****************************************************************************/
QStringList Picviz::PVView::get_axes_names_list()
{
	return axes_combination.get_axes_names_list();
}

/******************************************************************************
 *
 * Picviz::PVView::get_axis_name
 *
 *****************************************************************************/
QString Picviz::PVView::get_axis_name(PVCol index) const
{
	PVAxis const& axis = axes_combination.get_axis(index);
	return axis.get_name();
}

QString Picviz::PVView::get_axis_type(PVCol index) const
{
	PVAxis const& axis = axes_combination.get_axis(index);
	return axis.get_type();
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
PVCore::PVColor Picviz::PVView::get_color_in_output_layer(PVRow index)
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
	return plotted->get_column_count();
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
QString Picviz::PVView::get_data(PVRow row, PVCol column)
{
	PVCol real_index = axes_combination.get_axis_column_index_fast(column);

	return get_qtnraw_parent().at(row, real_index).get_qstr();
}

PVCore::PVUnicodeString const& Picviz::PVView::get_data_unistr(PVRow row, PVCol column)
{
	PVCol real_index = axes_combination.get_axis_column_index_fast(column);
	return get_rushnraw_parent().at_unistr(row, real_index);
}

/******************************************************************************
 *
 * Picviz::PVView::get_real_axis_index
 *
 *****************************************************************************/
PVCol Picviz::PVView::get_real_axis_index(PVCol col)
{
	return axes_combination.get_axis_column_index(col);
}

/******************************************************************************
 *
 * Picviz::PVView::get_data
 *
 *****************************************************************************/
QString Picviz::PVView::get_data_raw(PVRow row, PVCol column)
{
	return get_qtnraw_parent().at(row, column).get_qstr();
}

/******************************************************************************
 *
 * Picviz::PVView::get_floating_selection
 *
 *****************************************************************************/
Picviz::PVSelection &Picviz::PVView::get_floating_selection()
{
	return floating_selection;
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
int Picviz::PVView::get_layer_stack_layer_n_locked_state(int n)
{
	PVLayer &layer = layer_stack.get_layer_n(n);
	return layer.get_locked();
}

/******************************************************************************
 *
 * Picviz::PVView::get_layer_stack_layer_n_name
 *
 *****************************************************************************/
QString Picviz::PVView::get_layer_stack_layer_n_name(int n)
{
	PVLayer &layer = layer_stack.get_layer_n(n);
	return layer.get_name();
}

/******************************************************************************
 *
 * Picviz::PVView::get_layer_stack_layer_n_visible_state
 *
 *****************************************************************************/
int Picviz::PVView::get_layer_stack_layer_n_visible_state(int n)
{
	PVLayer &layer = layer_stack.get_layer_n(n);
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
 * Picviz::PVView::get_mapped_parent
 *
 *****************************************************************************/
Picviz::PVMapped* Picviz::PVView::get_mapped_parent()
{
	return plotted->get_mapped_parent();
}

const Picviz::PVMapped* Picviz::PVView::get_mapped_parent() const
{
	return plotted->get_mapped_parent();
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
int Picviz::PVView::get_original_axes_count()
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
 * Picviz::PVView::get_plotted_parent
 *
 *****************************************************************************/
Picviz::PVPlotted* Picviz::PVView::get_plotted_parent()
{
	return plotted;
}

const Picviz::PVPlotted* Picviz::PVView::get_plotted_parent() const
{
	return plotted;
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
 * Picviz::PVView::get_qtnraw_parent
 *
 *****************************************************************************/
PVRush::PVNraw::nraw_table& Picviz::PVView::get_qtnraw_parent()
{
	return plotted->get_qtnraw();
}

const PVRush::PVNraw::nraw_table& Picviz::PVView::get_qtnraw_parent() const
{
	return plotted->get_qtnraw();
}

/******************************************************************************
 *
 * Picviz::PVView::get_rushnraw_parent
 *
 *****************************************************************************/
PVRush::PVNraw& Picviz::PVView::get_rushnraw_parent()
{
	return plotted->get_mapped_parent()->get_source_parent()->get_rushnraw();
}

const PVRush::PVNraw& Picviz::PVView::get_rushnraw_parent() const
{
	return plotted->get_mapped_parent()->get_source_parent()->get_rushnraw();
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

/******************************************************************************
 *
 * Picviz::PVView::get_root
 *
 *****************************************************************************/
Picviz::PVRoot* Picviz::PVView::get_root()
{
	return plotted->get_root_parent();
}

/******************************************************************************
 *
 * Picviz::PVView::get_row_count
 *
 *****************************************************************************/
PVRow Picviz::PVView::get_row_count() const
{
	return plotted->get_row_count();
}


/******************************************************************************
 *
 * Picviz::PVView::get_source_parent
 *
 *****************************************************************************/
Picviz::PVSource* Picviz::PVView::get_source_parent()
{
	return plotted->get_source_parent();
}

const Picviz::PVSource* Picviz::PVView::get_source_parent() const
{
	return plotted->get_source_parent();
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
	if ( new_index != active_axis ) {
		axes_combination.move_axis_to_new_position(active_axis, new_index);
		active_axis = new_index;

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
	PVRow i;
	PVRow i_max = row_count;
	float f_imax = (float) i_max;

	/* We compute the real_output_selection */
	eventline.selection_A2B_filter(post_filter_layer.get_selection(), real_output_selection);

	/* We refresh the nu_selection */
	nu_selection = ~layer_stack_output_layer.get_selection();
	//layer_stack_output_layer->selection.A2B_inverse(nu_selection);
	
	nu_selection |= real_output_selection;
//	nu_selection.AB2A_or(real_output_selection);

	PVLinesProperties& out_lps = output_layer.get_lines_properties();
	PVLinesProperties const& post_lps = post_filter_layer.get_lines_properties();
	/* We are now able to process the lines_properties */
	for ( i=0; i<i_max; i++) {
		/* We check if the line is selected at the end of the process */
		PVCore::PVColor& out_lp = out_lps.get_line_properties(i);
		PVCore::PVColor const& post_lp = post_lps.get_line_properties(i);
		if (real_output_selection.get_line(i)) {
			/* It is selected, so we copy it's line properties */
			out_lps.get_line_properties(i) = post_lp;
			/* ... and set it's z_level */
			z_level_array.get_value(i) = layer_stack.get_lia().get_value(i) + (f_imax-(float)i)/f_imax;
			//z_level_array.get_value(i) = layer_stack.get_lia().get_value(i) + ((float)i)/f_imax;
		} else {
			/* It is not selected in the end, so we check if it was available in the beginning */
			if (layer_stack_output_layer.get_selection().get_line(i)) {
				/* The line was available, but is unselected */
				out_lp.r() = post_lp.r()/2;
			  	out_lp.g() = post_lp.g()/2;
			  	out_lp.b() = post_lp.b()/2;
				/* We set it's z_level */
				z_level_array.get_value(i) = (f_imax-(float)i)/f_imax - 1.0f;
				//z_level_array.get_value(i) = (float)i/f_imax - 1.0f;
			} else {
				/* The line is a zombie line */
				out_lp = default_zombie_line_properties;
				/* We set it's z_level */
				z_level_array.get_value(i) = (f_imax-(float)i)/f_imax - 2.0f;
				//z_level_array.get_value(i) = (float)i/f_imax - 2.0f;
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
void Picviz::PVView::process_from_layer_stack()
{
	tbb::tick_count start = tbb::tick_count::now();

	/* We start by reprocessing the layer_stack */
	process_layer_stack();
	process_selection();
	process_filter();
	process_eventline();
	process_visibility();

	tbb::tick_count end = tbb::tick_count::now();
	PVLOG_INFO("(Picviz::PVView::process_from_layer_stack) function took %0.4f seconds.\n", (end-start).seconds());
}

/******************************************************************************
 *
 * Picviz::PVView::process_from_selection
 *
 *****************************************************************************/
void Picviz::PVView::process_from_selection()
{
	PVLOG_DEBUG("Picviz::PVView::%s\n",__FUNCTION__);
	process_selection();
	process_filter();
	process_eventline();
	process_visibility();
        
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
			pre_filter_layer.get_selection() = floating_selection | volatile_selection;
//			floating_selection.AB2C_or(volatile_selection, pre_filter_layer->selection);
			break;

		case Picviz::PVStateMachine::AREA_MODE_SUBSTRACT_VOLATILE:
			pre_filter_layer.get_selection() = floating_selection - volatile_selection ;
			//floating_selection.AB2C_substraction(volatile_selection, pre_filter_layer->selection);
			break;

		case Picviz::PVStateMachine::AREA_MODE_INTERSECT_VOLATILE:
			pre_filter_layer.get_selection() = floating_selection & volatile_selection;
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
			output_layer.get_selection() |= ~layer_stack_output_layer.get_selection();
		}
	}
}

/******************************************************************************
 *
 * Picviz::PVView::selection_A2B_select_with_square_area
 *
 *****************************************************************************/
void Picviz::PVView::selection_A2B_select_with_square_area(PVSelection &a, PVSelection &b)
{
	int line_index;

	int axes_count;
	float axis_left;
	float axis_right;
	//float axis_pos;
	int delta_inner_absciss;
	float inner_absciss_max;
	float inner_absciss_min;
	float inner_absciss_second;
	float k;
	const float *plotted_array;
	int plotted_column_count;
	//int res_count;
	int row_count;
	const float *temp_pointer_in_array;
	//float x, y, xb, yb;
	//float x_left, x_right;
	float xmin, xmax;
	float y_al;
	float y_ar;
	float y_iamax;
	float y_iamin;
	float ymin, ymax;
        

	/* We set the axes_count for further reference */
	axes_count = axes_combination.get_axes_count();

	/* We need a fast reference to the array of floats in the plotted */
	plotted_array = &(plotted->get_table().at(0));
	/* We set the plotted_column_count for further reference */
	plotted_column_count = plotted->get_column_count();
	/* We set the row_count of that view, for later usage */
	row_count = plotted->get_row_count();
	/* We reset the selection b */
	b.select_none();

	/* We compute the xmin, xmax, ymin, ymax of the square area */
	xmin = picviz_min(square_area.get_start_x(), square_area.get_end_x());
	xmax = picviz_max(square_area.get_start_x(), square_area.get_end_x());
	ymin = picviz_min(square_area.get_start_y(), square_area.get_end_y());
	ymax = picviz_max(square_area.get_start_y(), square_area.get_end_y());

	/* if the square area is at the left of the first axis or at the right of the last axis, do nothing */
	if ((xmax < 0.0) || (axes_count -1 < xmin)) {
		return;
	}

	/* We compute the position of the first axis before xmin */
	/* ... and the one juste after and in the square area */
	if (xmin < 0) {
		axis_left = 0;
		inner_absciss_min = 0;
	}
	else {
		axis_left = floorf(xmin);
		inner_absciss_min = axis_left + 1;
	}
	/* We do the some on the right side */
	if (axes_count - 1 <= xmax) {
		axis_right = (float)(axes_count -1);
		inner_absciss_max = axis_right;
	}
	else {
		axis_right = floorf(xmax + 1);
		inner_absciss_max = picviz_min(axes_count -1, axis_right - 1);
	}

	assert(axis_left >= 0 && axis_left < axes_count);
	assert(axis_right >= 0 && axis_right < axes_count);

	/* We compute the distance between the two inner_absciss */
	delta_inner_absciss = (int)(inner_absciss_max - inner_absciss_min);

	/* We switch according to the presence or not of axis in the square area */
	switch (delta_inner_absciss) {
		case -1:
			/* we check all lines */
			for ( line_index = 0; line_index < row_count; line_index++ ) {
				/* in fact, we first test that the given line
				is selected in the layer_stack_output_layer */
				if (a.get_line(line_index)) {
					temp_pointer_in_array = plotted_array + line_index * plotted_column_count;

					y_al = temp_pointer_in_array[axes_combination.get_axis_column_index_fast(axis_left)];
					y_ar = temp_pointer_in_array[axes_combination.get_axis_column_index_fast(axis_right)];
					/* first case : left value is under ymin */
					if (y_al < ymin) {
						/* we check the segment with (xmax, ymin) and (xmin, ymax) */
						if ((y_ar * (xmax - axis_left) - ymin + y_al * (axis_right - xmax)) > 0 &&
								(y_ar * (xmin - axis_left) - ymax + y_al * (axis_right - xmin)) < 0) {
							b.set_bit_fast(line_index);
						}
					/* second case : left value is over ymax */
					} else if (y_al > ymax) {
						/* we check the segment with (xmax, ymax) and (xmin, ymin)*/
						if ((y_ar * (xmax - axis_left) - ymax + y_al * (axis_right - xmax)) < 0 &&
								(y_ar * (xmin - axis_left) - ymin + y_al * (axis_right - xmin)) > 0) {
							b.set_bit_fast(line_index);
						}
					/* last case : left value is in between ymin and ymax */
					} else {
						/* we check the segment with (xmin, ymax) and (xmin, ymin)*/
						if ((y_ar * (xmin - axis_left) - ymax + y_al * (axis_right - xmin)) < 0 &&
								(y_ar * (xmin - axis_left) - ymin + y_al * (axis_right - xmin)) > 0) {
							b.set_bit_fast(line_index);
						}
					}
				}
			}
			break;

		case 0:
			/* we check all lines */
			for ( line_index = 0; line_index < row_count; line_index++ ) {
				/* in fact, we first test that the given line
				is selected in the input_layer */
				if (a.get_line(line_index)) {
					temp_pointer_in_array = plotted_array + line_index*plotted_column_count;

					y_iamin = temp_pointer_in_array[axes_combination.get_axis_column_index_fast(inner_absciss_min)];
					/* first case : first inner value is under ymin */
					if (y_iamin < ymin) {
						/* We must no look at the left side if we are left-most */
						if (inner_absciss_min != 0) {
							y_al = temp_pointer_in_array[axes_combination.get_axis_column_index_fast(axis_left)];
							/* we check the segment before the first inner_absciss */
							if ((y_iamin * (xmin - axis_left) - ymin + y_al * (inner_absciss_min - xmin)) > 0) {
								b.set_bit_fast(line_index);
								continue;
							}
						}
						/* We must no look at the right side if we are left-most */
						if (inner_absciss_max != axes_count - 1) {
							/* we check the segment after the last inner_absciss */
							y_iamax = temp_pointer_in_array[axes_combination.get_axis_column_index_fast(inner_absciss_max)];
							y_ar = temp_pointer_in_array[axes_combination.get_axis_column_index_fast(axis_right)];
							if ((y_ar * (xmax - inner_absciss_max) - ymin + y_iamax * (axis_right - xmax)) > 0) {
								b.set_bit_fast(line_index);
								continue;
							}
						}
					/* second case : first inner value is over ymax */
					} else if (y_iamin > ymax) {
						/* We must no look at the left side if we are left-most */
						if (inner_absciss_min != 0) {
							y_al = temp_pointer_in_array[axes_combination.get_axis_column_index_fast(axis_left)];
							/* we check the segment before the first inner_absciss */
							if ((y_iamin * (xmin - axis_left) - ymax + y_al * (inner_absciss_min - xmin)) < 0) {
								b.set_bit_fast(line_index);
								continue;
							}
						}
						/* We must no look at the right side if we are left-most */
						if (inner_absciss_max != axes_count - 1) {
							/* we check the segment after the last inner_absciss */
							y_iamax = temp_pointer_in_array[axes_combination.get_axis_column_index_fast(inner_absciss_max)];
							y_ar = temp_pointer_in_array[axes_combination.get_axis_column_index_fast(axis_right)];
							if ((y_ar * (xmax - inner_absciss_max) - ymax + y_iamax * (axis_right - xmax)) < 0) {
								b.set_bit_fast(line_index);
							}
						}
					} else {
						b.set_bit_fast(line_index);
					}
				}
			}
			break;

		default:
			inner_absciss_second = inner_absciss_min + 1;
			/* we check all lines */
			for (line_index = 0; line_index < row_count; line_index++) {
				/* in fact, we first test that the given line
				is selected in the layer_stack_output_layer */
				if (a.get_line(line_index)) {
					temp_pointer_in_array = plotted_array + line_index*plotted_column_count;

					y_iamin = temp_pointer_in_array[axes_combination.get_axis_column_index_fast(inner_absciss_min)];
					/* first case : first inner value is under ymin */
					if (y_iamin < ymin) {
						/* we check if further values goes higher than ymin */
						for (k = inner_absciss_second; k <= inner_absciss_max; k++) {
							if (temp_pointer_in_array[axes_combination.get_axis_column_index_fast(k)] >= ymin) {
								b.set_bit_fast(line_index);
								goto end_loop;
							}
						}
						/* We must no look at the left side if we are left-most */
						if (inner_absciss_min != 0) {
							/* we check the segment before the first inner_absciss */
							y_al = temp_pointer_in_array[axes_combination.get_axis_column_index_fast(axis_left)];
							if ((y_iamin * (xmin-axis_left) - ymin + y_al * (inner_absciss_min - xmin)) > 0){
								b.set_bit_fast(line_index);
								continue;
							}
						}
						/* We must no look at the right side if we are left-most */
						if (inner_absciss_max != axes_count - 1) {
							/* we check the segment after the last inner_absciss */
							y_iamax = temp_pointer_in_array[axes_combination.get_axis_column_index_fast(inner_absciss_max)];
							y_ar = temp_pointer_in_array[axes_combination.get_axis_column_index_fast(axis_right)];
							if ((y_ar * (xmax - inner_absciss_max) - ymin + y_iamax * (axis_right - xmax)) > 0) {
								b.set_bit_fast(line_index);
								continue;
							}
						}
					/* second case : first inner value is over ymax */
					} else if ( y_iamin > ymax) {
						/* we check if further values goes lower than ymax */
						for (k = inner_absciss_second; k <= inner_absciss_max; k++) {
							if ( temp_pointer_in_array[axes_combination.get_axis_column_index_fast(k)] <= ymax ) {
								b.set_bit_fast(line_index);
								goto end_loop;
							}
						}
						/* We must no look at the left side if we are left-most */
						if (inner_absciss_min != 0) {
							/* we check the segment before the first inner_absciss */
							y_al = temp_pointer_in_array[axes_combination.get_axis_column_index_fast(axis_left)];
							if ((y_iamin * (xmin - axis_left) - ymax + y_al * (inner_absciss_min - xmin)) < 0) {
								b.set_bit_fast(line_index);
								continue;
							}
						}
						/* We must no look at the right side if we are left-most */
						if (inner_absciss_max != axes_count - 1) {
							/* we check the segment after the last inner_absciss */
							y_iamax = temp_pointer_in_array[axes_combination.get_axis_column_index_fast(inner_absciss_max)];

							y_ar = temp_pointer_in_array[axes_combination.get_axis_column_index_fast(axis_right)];
							if ((y_ar*(xmax-inner_absciss_max) - ymax + y_iamax*(axis_right - xmax)) < 0) {
								b.set_bit_fast(line_index);
							}
						}
					} else {
						b.set_bit_fast(line_index);
					}
				}
end_loop:;
			}
			break;
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
		active_axis = 0;
	} else {
		axes_count = axes_combination.get_axes_count();
		if ( closest_int >= axes_count ) {
			/* We set the rightmost AXIS as active */
			active_axis = axes_count - 1;
		} else {
			/* we can safely set the active AXIS to the closest_int */
			active_axis = closest_int;
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
void Picviz::PVView::set_color_on_active_layer(unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
	/* VARIABLES */
// TOCHECK:	PVLayer &active_layer = layer_stack.layer_get_by_index(0);
	PVLayer &active_layer = layer_stack.get_layer_n(0);

	// picviz_lines_properties_selection_set_rgba(active_layer->lines_properties, floating_selection, row_count, r, g, b, a);
	active_layer.get_lines_properties().selection_set_rgba(floating_selection, row_count, r, g, b, a);
// TOCHECK:	active_layer->lines_properties->selection_set_rgba(floating_selection, row_count, r, g, b, a);
}

/******************************************************************************
 *
 * Picviz::PVView::set_color_on_post_filter_layer
 *
 *****************************************************************************/
void Picviz::PVView::set_color_on_post_filter_layer(unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
	post_filter_layer.get_lines_properties().selection_set_rgba(post_filter_layer.get_selection(), row_count, r, g, b, a);
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
int Picviz::PVView::set_layer_stack_layer_n_name(int n, char *new_name)
{
	PVLayer &layer = layer_stack.get_layer_n(n);
	layer.set_name(new_name);
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
	state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE);
	volatile_selection = layer.get_selection();
	process_from_selection();
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
	return plotted->get_mapped_parent()->get_source_parent()->get_extractor();
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
	get_mapped_parent()->process_parent_source();
	get_plotted_parent()->process_from_parent_mapped(true);

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
	process_from_selection();
}

void Picviz::PVView::select_no_line()
{
	// Set square area mode w/ volatile
	state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE);
	volatile_selection.select_none();
	process_from_selection();
}

void Picviz::PVView::select_inv_lines()
{
	// Commit current volatile selection
	commit_volatile_in_floating_selection();
	// Set square area mode w/ volatile
	state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE);
	volatile_selection = ~floating_selection;
	process_from_selection();
}

QString Picviz::PVView::get_name() const
{
	return QString("mapped/plotted: %1/%2").arg(get_mapped_parent()->get_name()).arg(get_plotted_parent()->get_name());
}

QString Picviz::PVView::get_window_name() const
{
	QString ret = get_source_parent()->get_window_name() + " | ";
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

// Load/save and serialization
void Picviz::PVView::serialize_write(PVCore::PVSerializeObject& so)
{
	so.object("layer-stack", layer_stack, "Layers", true);
	so.object("axes-combination", axes_combination, "Axes combination", true);
}

void Picviz::PVView::serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	if (!so.object("layer-stack", layer_stack, "Layers", true)) {
		// If no layer stack, reset all layers so that we have one :)
		reset_layers();
	}
	so.object("axes-combination", axes_combination, "Axes combination", true);
}
