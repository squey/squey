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
#include <inendi/selection.h>

#include <inendi/index-array.h>



/******************************************************************************
 ******************************************************************************
 *
 * CREATOR/DESTRUCTOR
 *
 ******************************************************************************
 *****************************************************************************/

/******************************************************************************
 *
 * inendi_index_array_new
 * WARNING !
 * the array is NOT initialized
 *
 *****************************************************************************/
inendi_index_array_t *inendi_index_array_new(int initial_row_count)
{
	inendi_index_array_t *ia;

	ia = (inendi_index_array_t *)malloc(sizeof(inendi_index_array_t));
	if (!ia) {
		PVLOG_ERROR("Cannot allocate index-array!\n");
		return NULL;
	}

	ia->array = (int *)calloc(INENDI_INDEX_ARRAY_MAX_SIZE, sizeof(int));
	if ( ! ia->array ) {
		PVLOG_ERROR("Cannot allocate array for index-array!\n");
		return NULL;
	}

	if ( (0<=initial_row_count) && (initial_row_count<=INENDI_INDEX_ARRAY_MAX_SIZE) ) {
		ia->row_count = initial_row_count;
	} else {
		PVLOG_ERROR("Cannot set row_count while creating inendi_index_array because it is out of range!\n");
	}
	
	ia->index_count = 0;

	return ia;
}



/******************************************************************************
 *
 * inendi_index_array_destroy
 *
 *****************************************************************************/
void inendi_index_array_destroy(inendi_index_array_t *ia)
{
	free(ia->array);
	free(ia);
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
 * inendi_index_array_get_row_count
 *
 *****************************************************************************/
// int inendi_index_array_get_row_count(inendi_index_array_t *ia)
// {
// 	return ia->row_count;
// }
// 


/******************************************************************************
 *
 * inendi_index_array_get_index_count
 *
 *****************************************************************************/
int inendi_index_array_get_index_count(inendi_index_array_t *ia)
{
	return ia->index_count;
}



/******************************************************************************
 *
 * inendi_index_array_set_row_count
 *
 *****************************************************************************/
// void inendi_index_array_set_row_count(inendi_index_array_t *ia, int new_row_count)
// {
// 	if ( (0<=new_row_count) && (new_row_count<=INENDI_INDEX_ARRAY_MAX_SIZE) ) {
// 		ia->row_count = new_row_count;
// 	} else {
// 		PVLOG_ERROR("Cannot inendi_index_array_set_row_count because row_count is out of range!\n");
// 	}
// }



/******************************************************************************
 *
 * inendi_index_array_set_from_selection
 * WARNING !
 * it is up to you to check that ia->row_count is meaningfull with
 *  the given selection
 *
 *****************************************************************************/
void inendi_index_array_set_from_selection(inendi_index_array_t *ia, inendi_selection_t *selection)
{
	int index = 0;
	int k;
	
	for ( k=0; k<ia->row_count; k++) {
		if (inendi_selection_get_line(selection, k)) {
			ia->array[index] = k;
			index++;
		}
	}

	ia->index_count = index;
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
 * inendi_index_array_reset
 *
 *****************************************************************************/
// void inendi_index_array_reset_linear(inendi_index_array_t *ia)
// {
// 	int k;
// 
// 	for (k = 0; k < ia->row_count; k++) {
// 		ia->array[k] = k;
// 	}
// 
// 	ia->index_count = ia->row_count;
// }
// 









