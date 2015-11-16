/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <stdio.h>
#include <stdlib.h>

#include <pvkernel/core/general.h>

#include <inendi/general.h>
#include <inendi/layer-index-array.h>
#include <inendi/selection.h>




/******************************************************************************
 ******************************************************************************
 *
 * CREATOR/DESTRUCTOR
 *
 ******************************************************************************
 *****************************************************************************/

/******************************************************************************
 *
 * inendi_layer_index_array_new
 * WARNING !
 * the array is NOT initialized
 *
 *****************************************************************************/
inendi_layer_index_array_t *inendi_layer_index_array_new(int initial_row_count)
{
	inendi_layer_index_array_t *lia;

	lia = (inendi_layer_index_array_t *)malloc(sizeof(inendi_layer_index_array_t));
	if (!lia) {
		PVLOG_ERROR("Cannot allocate layer_index_array!\n");
		return NULL;
	}

	lia->array = (int *)calloc(INENDI_LAYER_INDEX_ARRAY_MAX_SIZE, sizeof(int));
	if ( ! lia->array ) {
		PVLOG_ERROR("Cannot allocate array for layer_index_array!\n");
		return NULL;
	}

	if ( (0<=initial_row_count) && (initial_row_count<=INENDI_LAYER_INDEX_ARRAY_MAX_SIZE) ) {
		lia->row_count = initial_row_count;
	} else {
		PVLOG_ERROR("Cannot set row_count while creating inendi_layer_index_array because it is out of range!\n");
	}

	lia->index_count = 0;

	return lia;
}



/******************************************************************************
 *
 * inendi_layer_index_array_destroy
 *
 *****************************************************************************/
void inendi_layer_index_array_destroy(inendi_layer_index_array_t *lia)
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
 * inendi_layer_index_array_get_row_count
 *
 *****************************************************************************/
int inendi_layer_index_array_get_row_count(inendi_layer_index_array_t *lia)
{
	return lia->row_count;
}



/******************************************************************************
 *
 * inendi_layer_index_array_get_index_count
 *
 *****************************************************************************/
int inendi_layer_index_array_get_index_count(inendi_layer_index_array_t *lia)
{
	return lia->index_count;
}



/******************************************************************************
 *
 * inendi_layer_index_array_set_row_count
 *
 *****************************************************************************/
void inendi_layer_index_array_set_row_count(inendi_layer_index_array_t *lia, int new_row_count)
{
	if ( (0<=new_row_count) && (new_row_count<=INENDI_LAYER_INDEX_ARRAY_MAX_SIZE) ) {
		lia->row_count = new_row_count;
	} else {
		PVLOG_ERROR("Cannot inendi_layer_index_array_set_row_count because row_count is out of range!\n");
	}
}


/******************************************************************************
 *
 * inendi_layer_index_array_get_value
 *
 *****************************************************************************/
int inendi_layer_index_array_get_value(inendi_layer_index_array_t *lia, int row_index)
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
 * inendi_layer_index_array_reset
 *
 *****************************************************************************/
void inendi_layer_index_array_initialize(inendi_layer_index_array_t *lia)
{
	int k;

	for (k = 0; k < lia->row_count; k++) {
		lia->array[k] = 0;
	}

	lia->index_count = 0;
}










