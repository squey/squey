//! \file PVLayerStack.cpp
//! $Id: PVLayerStack.cpp 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011


#include <picviz/PVLayer.h>
#include <picviz/PVLayerStack.h>
#include <picviz/PVSelection.h>




/******************************************************************************
 *
 * Picviz::PVLayerStack::PVLayerStack
 *
 *****************************************************************************/
Picviz::PVLayerStack::PVLayerStack(PVRow row_count) :
	lia(row_count)
{
	layer_count = 0;
	selected_layer_index = -1;
}

/******************************************************************************
 *
 * Picviz::PVLayerStack::append_layer
 *
 *****************************************************************************/
Picviz::PVLayer* Picviz::PVLayerStack::append_new_layer()
{
	/* We prepare the string for the name of the new layer */
	const QString new_layer_automatic_name = QString("New layer %1").arg(layer_count);
	return append_layer(PVLayer(new_layer_automatic_name));
}

/******************************************************************************
 *
 * Picviz::PVLayerStack::append_layer
 *
 *****************************************************************************/
Picviz::PVLayer* Picviz::PVLayerStack::append_layer(const PVLayer &layer)
{
	/* We test if we have not reached the maximal number of layers */
	if (layer_count < PICVIZ_LAYER_STACK_MAX_DEPTH-1) {
		table.append(layer);
		selected_layer_index = layer_count;
		layer_count += 1;

		/* FIXME! This is before we do something more clever... */
		update_layer_index_array_completely();

		return &table.last();
	}
	// FIXME: should have an exception here, that will be treated by the GUI !
	return NULL;
}

/******************************************************************************
 *
 * Picviz::PVLayerStack::append_new_layer_from_selection_and_lines_properties
 *
 *****************************************************************************/
Picviz::PVLayer* Picviz::PVLayerStack::append_new_layer_from_selection_and_lines_properties(PVSelection const& selection, PVLinesProperties const& lines_properties)
{
	/* We prepare the string for the name of the new layer */
	QString new_layer_automatic_name = QString("New layer %1").arg(layer_count);
	return append_layer(PVLayer(new_layer_automatic_name, selection, lines_properties));
}


/******************************************************************************
 *
 * Picviz::PVLayerStack::delete_by_index
 *
 *****************************************************************************/
void Picviz::PVLayerStack::delete_by_index(int index)
{
	if ((0 < index) && (index < layer_count)) {
		table.removeAt(index);
		layer_count--;

		/* We set the selected layer according to the posibilities */
		if (layer_count == 0) {
			selected_layer_index = -1;
		} else {
			selected_layer_index = 0;
		}
		
		/* FIXME! This is before we do something more clever... */
		update_layer_index_array_completely();
	}
}

/******************************************************************************
 *
 * Picviz::PVLayerStack::delete_selected_layer
 *
 *****************************************************************************/
void Picviz::PVLayerStack::delete_selected_layer()
{
	delete_by_index(selected_layer_index);
}

/******************************************************************************
 *
 * Picviz::PVLayerStack::delete_all_layers
 *
 *****************************************************************************/
void Picviz::PVLayerStack::delete_all_layers()
{
	table.clear();
	layer_count = 0;
	selected_layer_index = -1;
	lia.initialize();
}

/**********************************************************************
*
* Picviz::PVLayerStack::move_layer_down
*
**********************************************************************/
void Picviz::PVLayerStack::move_layer_down(int index)
{
	if (( 0 < index ) && (index < layer_count)) {
		table.move(index, index - 1);
		
		/* FIXME! This is before we do something more clever... */
		update_layer_index_array_completely();
	}
}

/**********************************************************************
*
* Picviz::PVLayerStack::move_layer_up
*
**********************************************************************/
void Picviz::PVLayerStack::move_layer_up(int index)
{
	if ( index < (layer_count - 1)) {
		table.move(index, index + 1);

		/* FIXME! This is before we do something more clever... */
		update_layer_index_array_completely();
	}
}


/**********************************************************************
*
* Picviz::PVLayerStack::move_selected_layer_down
*
**********************************************************************/
void Picviz::PVLayerStack::move_selected_layer_down()
{
	if (selected_layer_index > 0) {
		move_layer_down(selected_layer_index);
		selected_layer_index  -= 1;
	}
}

/**********************************************************************
 *
 * Picviz::PVLayerStack::move_selected_layer_up
 *
 **********************************************************************/
void Picviz::PVLayerStack::move_selected_layer_up()
{
	if (selected_layer_index < layer_count - 1) {
		move_layer_up(selected_layer_index);
		selected_layer_index  += 1;
	}
}

/**********************************************************************
 *
 * Picviz::PVLayerStack::process
 *
 **********************************************************************/
void Picviz::PVLayerStack::process(PVLayer &output_layer, PVRow row_count)
{
	/******************************
	* Preparation
	******************************/
	/* We prepare a pointer to the layer of the LS being processed */
	PVLayer *layer_being_processed;
	/* We prepare a temporary selection, needed in our computations */
	PVSelection temp_selection;
	/* We store locally the layer-stack->layer_count and prepare a counter */
	int i;
	/* ... and the number of rows in the parent view */
	int view_number_of_rows;

	/******************************
	* Initializations
	******************************/
	/* It's time to erase the output_layer ! */
	output_layer.reset_to_empty_and_default_color();

	// /* We get the number of rows/events in the parent view */
	view_number_of_rows = row_count;

	/******************************
	* Main computation
	******************************/
	/* The layer-stack might be empty so we have to be carefull... */
	if ( layer_count > 0 ) {
		/* We process the layers from TOP to BOTTOM, that is, from
		*  the most visible to the less visible */
		for ( i=layer_count-1; i >= 0; i--) {
			/* We prepare a direct access to the layer we have to process */
			layer_being_processed = &(table[i]);
			/* We check if this layer is visible */
			if (layer_being_processed->get_visible()) {
				/* we compute the selection of lines present
				*  in the layer being processed but not yet
				*  in the output layer */
				temp_selection = layer_being_processed->get_selection() - output_layer.get_selection();
				//layer_being_processed->selection.AB2C_substraction(output_layer->selection, temp_selection);
				/* and we already update the selection in
				*  the output_layer */
				output_layer.get_selection() |= layer_being_processed->get_selection();
				//output_layer->selection.AB2A_or(layer_being_processed->selection);

				/* We copy in the output_layer the only new
				*  lines properties */
				// picviz_lines_properties_A2B_copy_restricted_by_selection_and_nelts(layer_being_processed->lines_properties, output_layer->lines_properties, temp_selection, view_number_of_rows);
				layer_being_processed->get_lines_properties().A2B_copy_restricted_by_selection_and_nelts(output_layer.get_lines_properties(), temp_selection, (pvcol)view_number_of_rows);
			}
		}
	}
}

/******************************************************************************
 *
 * update_layer_index_array_completely
 *
 * WARNING!
 *
 * This function, under normal circumstances, should not be use !
 *   The layer_index_array should be maintained up to date, step by step, by
 *   "tacking into account" the changes done in the layer_stack.
 *
 * This function is a starter...
 *
 *****************************************************************************/
/**********************************************************************
 *
 * Picviz::PVLayerStack::update_layer_index_array_completely
 *
 **********************************************************************/
void Picviz::PVLayerStack::update_layer_index_array_completely()
{
	/******************************
	* Preparation
	******************************/
	/* We prepare a reference to the layer of the LS being processed */
// 	PVLayer &layer_being_processed;
	/* We prepare the two selections needed in our algorithm */
	PVSelection done_selection;
	PVSelection temp_selection;
	/* We prepare a direct access to the number of layers, and 2 counters */
       	int i, k;

	/******************************
	* Main computation
	******************************/
	/* The layer-stack might be empty so we have to be carefull... */
	if ( layer_count > 0 ) {
		/* We process the layers from top to bottom */
		for ( i=layer_count-1; i >= 0; i--) {
			/* We prepare a direct access to the layer we have to process */
			PVLayer &layer_being_processed = table[i];
			/* We check if the layer is visible */
			if (layer_being_processed.get_visible()) {
				/* We remove from temp selection the lines already encountered */
				temp_selection = layer_being_processed.get_selection();
				temp_selection -= done_selection;
				/* We add to done_selection the lines that we are about to process */
				done_selection |= layer_being_processed.get_selection();
				for (k = 0; k < lia.get_row_count(); k++) {
					if (temp_selection.get_line(k)) {
						lia.set_value(k, i + 1);
					}
				}
			}
		}

		/* We set to 0 the lines that never appeared in any layer */
		for (k = 0; k < lia.get_row_count(); k++) {
			if ( ! done_selection.get_line(k) ) {
				lia.set_value(k, 0);
			}
		}
	} else {
		/* We zero every line since there are no layers */
		for (k = 0; k < lia.get_row_count(); k++) {
			lia.set_value(k, 0);
		}
	}

	/******************************
	* Garbage collection
	******************************/
}



