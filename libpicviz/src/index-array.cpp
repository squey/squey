//! \file index-array.cpp
//! $Id: index-array.cpp 2548 2011-05-03 15:46:52Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <stdio.h>
#include <stdlib.h>

#include <pvcore/general.h>

#include <picviz/general.h>
#include <picviz/selection.h>

#include <picviz/index-array.h>



/******************************************************************************
 ******************************************************************************
 *
 * CREATOR/DESTRUCTOR
 *
 ******************************************************************************
 *****************************************************************************/

/******************************************************************************
 *
 * picviz_index_array_new
 * WARNING !
 * the array is NOT initialized
 *
 *****************************************************************************/
picviz_index_array_t *picviz_index_array_new(int initial_row_count)
{
	picviz_index_array_t *ia;

	ia = (picviz_index_array_t *)malloc(sizeof(picviz_index_array_t));
	if (!ia) {
		PVLOG_ERROR("Cannot allocate index-array!\n");
		return NULL;
	}

	ia->array = (int *)calloc(PICVIZ_INDEX_ARRAY_MAX_SIZE, sizeof(int));
	if ( ! ia->array ) {
		PVLOG_ERROR("Cannot allocate array for index-array!\n");
		return NULL;
	}

	if ( (0<=initial_row_count) && (initial_row_count<=PICVIZ_INDEX_ARRAY_MAX_SIZE) ) {
		ia->row_count = initial_row_count;
	} else {
		PVLOG_ERROR("Cannot set row_count while creating picviz_index_array because it is out of range!\n");
	}
	
	ia->index_count = 0;

	return ia;
}



/******************************************************************************
 *
 * picviz_index_array_destroy
 *
 *****************************************************************************/
void picviz_index_array_destroy(picviz_index_array_t *ia)
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
 * picviz_index_array_get_row_count
 *
 *****************************************************************************/
// int picviz_index_array_get_row_count(picviz_index_array_t *ia)
// {
// 	return ia->row_count;
// }
// 


/******************************************************************************
 *
 * picviz_index_array_get_index_count
 *
 *****************************************************************************/
int picviz_index_array_get_index_count(picviz_index_array_t *ia)
{
	return ia->index_count;
}



/******************************************************************************
 *
 * picviz_index_array_set_row_count
 *
 *****************************************************************************/
// void picviz_index_array_set_row_count(picviz_index_array_t *ia, int new_row_count)
// {
// 	if ( (0<=new_row_count) && (new_row_count<=PICVIZ_INDEX_ARRAY_MAX_SIZE) ) {
// 		ia->row_count = new_row_count;
// 	} else {
// 		PVLOG_ERROR("Cannot picviz_index_array_set_row_count because row_count is out of range!\n");
// 	}
// }



/******************************************************************************
 *
 * picviz_index_array_set_from_selection
 * WARNING !
 * it is up to you to check that ia->row_count is meaningfull with
 *  the given selection
 *
 *****************************************************************************/
void picviz_index_array_set_from_selection(picviz_index_array_t *ia, picviz_selection_t *selection)
{
	int index = 0;
	int k;
	
	for ( k=0; k<ia->row_count; k++) {
		if (picviz_selection_get_line(selection, k)) {
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
 * picviz_index_array_reset
 *
 *****************************************************************************/
// void picviz_index_array_reset_linear(picviz_index_array_t *ia)
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









