/**
 * \file layer-index-array.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <stdio.h>
#include <stdlib.h>

#include <pvkernel/core/general.h>

#include <picviz/general.h>
#include <picviz/layer-index-array.h>
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
 * picviz_layer_index_array_new
 * WARNING !
 * the array is NOT initialized
 *
 *****************************************************************************/
picviz_layer_index_array_t *picviz_layer_index_array_new(int initial_row_count)
{
	picviz_layer_index_array_t *lia;

	lia = (picviz_layer_index_array_t *)malloc(sizeof(picviz_layer_index_array_t));
	if (!lia) {
		PVLOG_ERROR("Cannot allocate layer_index_array!\n");
		return NULL;
	}

	lia->array = (int *)calloc(PICVIZ_LAYER_INDEX_ARRAY_MAX_SIZE, sizeof(int));
	if ( ! lia->array ) {
		PVLOG_ERROR("Cannot allocate array for layer_index_array!\n");
		return NULL;
	}

	if ( (0<=initial_row_count) && (initial_row_count<=PICVIZ_LAYER_INDEX_ARRAY_MAX_SIZE) ) {
		lia->row_count = initial_row_count;
	} else {
		PVLOG_ERROR("Cannot set row_count while creating picviz_layer_index_array because it is out of range!\n");
	}

	lia->index_count = 0;

	return lia;
}



/******************************************************************************
 *
 * picviz_layer_index_array_destroy
 *
 *****************************************************************************/
void picviz_layer_index_array_destroy(picviz_layer_index_array_t *lia)
{
	free(lia->array);
	free(lia);
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
 * picviz_layer_index_array_get_row_count
 *
 *****************************************************************************/
int picviz_layer_index_array_get_row_count(picviz_layer_index_array_t *lia)
{
	return lia->row_count;
}



/******************************************************************************
 *
 * picviz_layer_index_array_get_index_count
 *
 *****************************************************************************/
int picviz_layer_index_array_get_index_count(picviz_layer_index_array_t *lia)
{
	return lia->index_count;
}



/******************************************************************************
 *
 * picviz_layer_index_array_set_row_count
 *
 *****************************************************************************/
void picviz_layer_index_array_set_row_count(picviz_layer_index_array_t *lia, int new_row_count)
{
	if ( (0<=new_row_count) && (new_row_count<=PICVIZ_LAYER_INDEX_ARRAY_MAX_SIZE) ) {
		lia->row_count = new_row_count;
	} else {
		PVLOG_ERROR("Cannot picviz_layer_index_array_set_row_count because row_count is out of range!\n");
	}
}


/******************************************************************************
 *
 * picviz_layer_index_array_get_value
 *
 *****************************************************************************/
int picviz_layer_index_array_get_value(picviz_layer_index_array_t *lia, int row_index)
{
	return lia->array[row_index];
}



/******************************************************************************
 ******************************************************************************
 *
 * SPECIFIC functions
 *
 ******************************************************************************
 *****************************************************************************/

/******************************************************************************
 *
 * picviz_layer_index_array_reset
 *
 *****************************************************************************/
void picviz_layer_index_array_initialize(picviz_layer_index_array_t *lia)
{
	int k;

	for (k = 0; k < lia->row_count; k++) {
		lia->array[k] = 0;
	}

	lia->index_count = 0;
}










