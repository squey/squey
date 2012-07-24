/**
 * \file layer-stack.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <stdio.h>
#include <stdlib.h>

#include <pvkernel/core/general.h>

#include <picviz/layer-stack.h>
#include <picviz/general.h>
#include <picviz/layer.h>
#include <picviz/selection.h>


/******************************************************************************
 ******************************************************************************
 *
 * CREATOR/DESTRUCTOR
 *
 ******************************************************************************
 *****************************************************************************/

/******************************************************************************
 *
 * picviz_layer_stack_new
 *
 *****************************************************************************/
picviz_layer_stack_t *picviz_layer_stack_new(PVRow row_count)
{
	picviz_layer_stack_t *layer_stack;
	
	layer_stack = (picviz_layer_stack_t *)malloc(sizeof(picviz_layer_stack_t));
	if (!layer_stack) {
		PVLOG_ERROR("Cannot allocate layer stack!\n");
		return NULL;
	}

	layer_stack->lia = picviz_layer_index_array_new(row_count);

	// if (view) {
	//   layer_stack->parent = view;
	//   layer_stack->lia = picviz_layer_index_array_new(picviz_view_get_row_count((picviz_view_t *)view));
	//   /* memset(layer_stack->lia, 0, picviz_view_get_row_count(view)); */
	// } else {
	//   layer_stack->lia = picviz_layer_index_array_new(1);
	// 	/* memset(layer_stack->lia, 0, 1); */
	// }

	picviz_layer_index_array_initialize(layer_stack->lia);
	
	layer_stack->layer_count = 0;
	layer_stack->selected_layer_index = -1;
	
	return layer_stack;
}





/******************************************************************************
 *
 * picviz_layer_stack_destroy
 *
 *****************************************************************************/
void picviz_layer_stack_destroy(picviz_layer_stack_t *layer_stack)
{
	int i;
	
	for ( i=0; i < layer_stack->layer_count; i++) {
		picviz_layer_destroy(layer_stack->table[i]);
	}

	picviz_layer_index_array_destroy(layer_stack->lia);
	
	free(layer_stack);
}





/******************************************************************************
 ******************************************************************************
 *
 * GET/SET functions
 *
 ******************************************************************************
 *****************************************************************************/

/******************************************************************************
 *
 * picviz_layer_stack_get_layer_count
 *
 *****************************************************************************/
int picviz_layer_stack_get_layer_count(picviz_layer_stack_t *layer_stack)
{
	return layer_stack->layer_count;
}



/******************************************************************************
 *
 * picviz_layer_stack_get_layer_n
 *
 *****************************************************************************/
picviz_layer_t *picviz_layer_stack_get_layer_n(picviz_layer_stack_t *layer_stack, int n)
{
	if ((0<=n) && (n<layer_stack->layer_count)) {
		return layer_stack->table[n];
	} else {
		PVLOG_ERROR("picviz_layer_stack_get_layer_n : layer index not in range!\n");
		return 0;
	}
}



/******************************************************************************
 *
 * picviz_layer_stack_get_selected_layer
 *
 *****************************************************************************/
picviz_layer_t *picviz_layer_stack_get_selected_layer(picviz_layer_stack_t *layer_stack)
{
	return layer_stack->table[layer_stack->selected_layer_index];
}



/******************************************************************************
 *
 * picviz_layer_stack_get_selected_layer_index
 *
 *****************************************************************************/
int picviz_layer_stack_get_selected_layer_index(picviz_layer_stack_t *layer_stack)
{
	return layer_stack->selected_layer_index;
}



/******************************************************************************
 *
 * picviz_layer_stack_set_selected_layer_index
 *
 *****************************************************************************/
void picviz_layer_stack_set_selected_layer_index(picviz_layer_stack_t *layer_stack, int index)
{
	if ((0 <= index) && (index < layer_stack->layer_count)) {
		layer_stack->selected_layer_index = index;
	}
}





/******************************************************************************
 ******************************************************************************
 *
 * GENERIC FUNCTIONS ON THE LAYER STACK
 *
 ******************************************************************************
 *****************************************************************************/

/******************************************************************************
 *
 * picviz_layer_stack_process
 *
 *****************************************************************************/
void picviz_layer_stack_process(picviz_layer_stack_t *layer_stack, picviz_layer_t *output_layer, PVRow row_count)
{
	/******************************
	* Preparation
	******************************/
	/* We prepare a temporary layer, needed in our computations */
	picviz_layer_t *temp_layer;
	/* We prepare a pointer to the layer of the LS being processed */
	picviz_layer_t *layer_being_processed;
	
	/* We prepare a temporary selection, needed in our computations */
	picviz_selection_t *temp_selection;

	/* We store locally the layer-stack->layer_count and prepare a counter */
       	int ls_layer_count, i;

	/* We prepare a direct access to the parent view ... */
	// picviz_view_t *view_parent;
	/* ... and the number of rows in the parent view */
	int view_number_of_rows;

	/******************************
	* Initializations
	******************************/
	/* It's time to erase the output_layer ! */
	picviz_layer_A2A_reset_to_empty_and_default_color(output_layer);

	/* Now we create the temp_layer */
	temp_selection = picviz_selection_new();

	/* We get the number of layer in the layer-stack */
	ls_layer_count = layer_stack->layer_count;

	/* We go up and cast for the parent view */
	// view_parent = (picviz_view_t *)layer_stack->PICVIZ_OBJECT_VIEW(parent);
	// /* We get the number of rows/events in the parent view */
	view_number_of_rows = row_count;
	
	/******************************
	* Main computation
	******************************/
	/* The layer-stack might be empty so we have to be carefull... */
	if ( ls_layer_count > 0 ) {
		/* We process the layers from TOP to BOTTOM, that is, from
		*  the most visible to the less visible */
		for ( i=ls_layer_count-1; i >= 0; i--) {
			/* We prepare a direct access to the layer we have to process */
			layer_being_processed = layer_stack->table[i];
			/* We check if this layer is visible */
			if (picviz_layer_get_visible(layer_being_processed)) {
				/* we compute the selection of lines present
				*  in the layer being processed but not yet
				*  in the output layer */
				picviz_selection_AB2C_substraction(layer_being_processed->selection, output_layer->selection, temp_selection);
				/* and we already update the selection in
				*  the output_layer */
				picviz_selection_AB2A_or(output_layer->selection, layer_being_processed->selection);

				/* We copy in the output_layer the only new
				*  lines properties */
				//picviz_lines_properties_A2B_copy_restricted_by_selection_and_nelts(layer_being_processed->lines_properties, output_layer->lines_properties, temp_selection, view_number_of_rows);
				//layer_being_processed->lines_properties->A2B_copy_restricted_by_selection_and_nelts(output_layer->lines_properties, temp_selection, (pvcol)view_number_of_rows);
			}
		}
	}

	picviz_selection_destroy(temp_selection);
}



/******************************************************************************
 *
 * picviz_layer_stack_update_layer_index_array_completely
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
void picviz_layer_stack_update_layer_index_array_completely(picviz_layer_stack_t *layer_stack)
{
	/******************************
	* Preparation
	******************************/
	/* We prepare a direct pointer to the lia */
	picviz_layer_index_array_t *lia;
	/* We prepare a pointer to the layer of the LS being processed */
	picviz_layer_t *layer_being_processed;
	/* We prepare pointers for the two selections needed in our algorithm */
	picviz_selection_t *done_selection;
	picviz_selection_t *temp_selection;
	/* We prepare a direct access to the number of layers, and 2 counters */
       	int ls_layer_count, i, k;

	/******************************
	* Initializations
	******************************/
	/* We set the pointer for a direct access to the lia */
	lia = layer_stack->lia;
	/* We create the  two needed selections */
	done_selection = picviz_selection_new();
	temp_selection = picviz_selection_new();
	/* We set the done_selection to an empty selection... */
	/* We compute and store locally the number of layers */
	ls_layer_count = layer_stack->layer_count;

	/******************************
	* Main computation
	******************************/
	/* The layer-stack might be empty so we have to be carefull... */
	if ( ls_layer_count > 0 ) {
		/* We process the layers from top to bottom */
		for ( i=ls_layer_count-1; i >= 0; i--) {
			/* We prepare a direct access to the layer we have to process */
			layer_being_processed = layer_stack->table[i];
			/* We check if the layer is visible */
			if (picviz_layer_get_visible(layer_being_processed)) {
				picviz_selection_AB2C_substraction(layer_being_processed->selection, done_selection, temp_selection);
				picviz_selection_AB2A_or(done_selection, layer_being_processed->selection);
				for (k = 0; k < lia->row_count; k++) {
					if (picviz_selection_get_line(temp_selection, k)) {
						lia->array[k] = i + 1;
					}
				}
			}
		}

		/* We set to 0 the lines that never appeared in any layer */
		for (k = 0; k < lia->row_count; k++) {
			if ( ! picviz_selection_get_line(done_selection, k) ) {
				lia->array[k] = 0;
			}
		}
	} else {
		/* We zero every line since there are no layers */
		for (k = 0; k < lia->row_count; k++) {
			lia->array[k] = 0;
		}
	}

	/******************************
	* Garbage collection
	******************************/
	/* We delete the two selection used during our computation */
	picviz_selection_destroy(done_selection);
	picviz_selection_destroy(temp_selection);
}




/******************************************************************************
 ******************************************************************************
 *
 * FUNCTIONS TO MANIPULATE THE LAYERS IN THE STACK
 *
 ******************************************************************************
 *****************************************************************************/


/******************************************************************************
 *
 * picviz_layer_stack_append_layer
 *
 *****************************************************************************/
void picviz_layer_stack_append_layer(picviz_layer_stack_t *layer_stack, picviz_layer_t *layer)
{
	/* We test if we have not reached the maximal number of layers */
	if (layer_stack->layer_count < PICVIZ_LAYER_STACK_MAX_DEPTH-1) {
		/* we store the pointer to the appended layer at the right place */
		layer_stack->table[layer_stack->layer_count] = layer;
		/* this appended layer is always the new selected one */
		layer_stack->selected_layer_index = layer_stack->layer_count;
		/* We increese by one the total number of layers */
		layer_stack->layer_count += 1;
	}

	/* FIXME! This is before we do something more clever... */
	picviz_layer_stack_update_layer_index_array_completely(layer_stack);
}



/******************************************************************************
 *
 * picviz_layer_stack_append_new_layer
 *
 *****************************************************************************/
void picviz_layer_stack_append_new_layer(picviz_layer_stack_t *layer_stack)
{
	/* We prepare the string for the name of the new layer */
	char new_layer_automatic_name[] = "New layer 256"; /* the 256 will be replaced later */
	sprintf(new_layer_automatic_name + 10, "%03d", layer_stack->layer_count);

	/* We test if we have not reached the maximal number of layers */
	if (layer_stack->layer_count < PICVIZ_LAYER_STACK_MAX_DEPTH-1) {
		/* we store the pointer to a newly created layer */
		layer_stack->table[layer_stack->layer_count] = picviz_layer_new(new_layer_automatic_name);
		/* this new layer is always the new selected one */
		layer_stack->selected_layer_index = layer_stack->layer_count;
		/* We increese by one the total number of layers */
		layer_stack->layer_count += 1;
	}

	/* FIXME! This is before we do something more clever... */
	picviz_layer_stack_update_layer_index_array_completely(layer_stack);
}



/******************************************************************************
 *
 * picviz_layer_stack_append_new_layer_from_layer
 *
 *****************************************************************************/
// void picviz_layer_stack_append_new_layer_from_layer(picviz_layer_stack_t *layer_stack, picviz_layer_t *layer)
// {
// 	int new_layer_index;
// 	picviz_layer_stack_append_new_layer(layer_stack);
// 	new_layer_index = layer_stack->layer_count - 1;
// 	picviz_layer_A2B_copy(layer, layer_stack->table[new_layer_index]);
// 
// 	/* If the is the first layer, it is selected by default */
// 	layer_stack->selected_layer_index = 0;
// 
// 	/* FIXME! This is before we do something more clever... */
// 	picviz_layer_stack_update_layer_index_array_completely(layer_stack);
// }



/******************************************************************************
 *
 * picviz_layer_stack_append_new_layer_from_selection_and_lines_properties
 *
 *****************************************************************************/
void picviz_layer_stack_append_new_layer_from_selection_and_lines_properties(picviz_layer_stack_t *layer_stack, picviz_selection_t *selection, Picviz::PVLinesProperties *lines_properties)
{
	int new_layer_index;
	picviz_layer_stack_append_new_layer(layer_stack);
	new_layer_index = layer_stack->layer_count - 1;
	picviz_selection_A2B_copy(selection, (layer_stack->table[new_layer_index])->selection);
	qDebug("%s: Double check this function for PVLineSProperties\n", __FUNCTION__);
	std::swap(layer_stack->table[new_layer_index]->lines_properties, lines_properties);
	// memcpy(layer_stack->table[new_layer_index]->lines_properties, lines_properties, sizeof(Picviz::PVLinesProperties));
	// picviz_lines_properties_A2B_copy(lines_properties, (layer_stack->table[new_layer_index])->lines_properties);

	/* FIXME! This is before we do something more clever... */
	picviz_layer_stack_update_layer_index_array_completely(layer_stack);
}



 /*****************************************************************************
 *
 * picviz_layer_stack_layer_delete_by_index
 *
 *****************************************************************************/
void picviz_layer_stack_layer_delete_by_index(picviz_layer_stack_t *layer_stack, int index)
{
	/* We prepare the counter for the loop */
	int i;

	/* We start by detroying the layer */
	picviz_layer_destroy(layer_stack->table[index]);

	/* We shift all remaining layers one level down */
	for ( i=index; i<(layer_stack->layer_count -1); i++) {
		layer_stack->table[i] = layer_stack->table[i+1];
	}

	/* We decrease by one the total number of layers */
	layer_stack->layer_count -= 1;
	/* We set the selected layer according to the posibilities */
	if (layer_stack->layer_count == 0) {
		layer_stack->selected_layer_index = -1;
	} else {
		layer_stack->selected_layer_index = 0;
	}

	/* FIXME! This is before we do something more clever... */
	picviz_layer_stack_update_layer_index_array_completely(layer_stack);
}


 /*****************************************************************************
 *
 * picviz_layer_stack_delete_selected_layer
 *
 *****************************************************************************/
void picviz_layer_stack_delete_selected_layer(picviz_layer_stack_t *layer_stack)
{
	picviz_layer_stack_layer_delete_by_index(layer_stack, layer_stack->selected_layer_index);
}



/******************************************************************************
 *
 * picviz_layer_stack_layer_get_by_index
 *
 *****************************************************************************/
picviz_layer_t *picviz_layer_stack_layer_get_by_index(picviz_layer_stack_t *layer_stack, int index)
{
	 return layer_stack->table[index];
}



/**********************************************************************
*
* picviz_layer_stack_layer_get_by_name
*
**********************************************************************/
// picviz_layer_t *picviz_layer_stack_layer_get_by_name(picviz_layer_stack_t *layer_stack, char *name)
// {
// 	int count;
// 
// 	for ( count = 0; count < layer_stack->layer_count; count++ ) {
// 		picviz_layer_t *layer = layer_stack->table[count];
// 		if (layer) {
// 			if (!strcmp(layer->name, name)) {
// 				return layer;
// 			}
// 		}
// 	}
// 
// 	return NULL;
// }




/**********************************************************************
*
* picviz_layer_stack_move_layer_down
*
**********************************************************************/
int picviz_layer_stack_move_layer_down(picviz_layer_stack_t *layer_stack, int index)
{
	if ( 0 < index ) {
		picviz_layer_t *temp;
		temp = layer_stack->table[index];
		layer_stack->table[index] = layer_stack->table[index-1];
		layer_stack->table[index-1] = temp;
	}

	/* FIXME! This is before we do something more clever... */
	picviz_layer_stack_update_layer_index_array_completely(layer_stack);
	
	return 0;
}


/**********************************************************************
*
* picviz_layer_stack_move_layer_up
*
**********************************************************************/
int picviz_layer_stack_move_layer_up(picviz_layer_stack_t *layer_stack, int index)
{
	if ( index < (layer_stack->layer_count - 1)) {
		picviz_layer_t *temp;
		temp = layer_stack->table[index];
		layer_stack->table[index] = layer_stack->table[index+1];
		layer_stack->table[index+1] = temp;
	}

	/* FIXME! This is before we do something more clever... */
	picviz_layer_stack_update_layer_index_array_completely(layer_stack);
	
	return 0;
}



/**********************************************************************
*
* picviz_layer_stack_move_selected_layer_down
*
**********************************************************************/
void picviz_layer_stack_move_selected_layer_down(picviz_layer_stack_t *layer_stack)
{
	if (layer_stack->selected_layer_index > 0) {
		picviz_layer_stack_move_layer_down(layer_stack, layer_stack->selected_layer_index);
		layer_stack->selected_layer_index  -= 1;
	}
}


/**********************************************************************
*
* picviz_layer_stack_move_selected_layer_up
*
**********************************************************************/
void picviz_layer_stack_move_selected_layer_up(picviz_layer_stack_t *layer_stack)
{
	if (layer_stack->selected_layer_index < (layer_stack->layer_count - 1)) {
		picviz_layer_stack_move_layer_up(layer_stack, layer_stack->selected_layer_index);
		layer_stack->selected_layer_index  += 1;
	}
}


void picviz_layer_stack_debug(picviz_layer_stack_t *layer_stack)
{
	int count;
	int i;
	picviz_layer_t *layer;
	
	printf("Layer Stack\n===========\n");
	count = picviz_layer_stack_get_layer_count(layer_stack);
	for (i=0; i<count; i++) {
		layer = picviz_layer_stack_get_layer_n(layer_stack, i);
		picviz_layer_debug(layer);
	}
}



// TO BE CODED !!!

/**********************************************************************
*
* picviz_layer_stack_layer_delete_by_name
*
**********************************************************************/
void picviz_layer_stack_delete_layer_by_name(picviz_layer_stack_t *stack, char *layer)
{

}

