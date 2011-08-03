//! \file z-level-array.cpp
//! $Id: z-level-array.cpp 2556 2011-05-03 19:47:41Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <stdio.h>
#include <stdlib.h>

#include <pvkernel/core/general.h>

#include <picviz/general.h>
#include <picviz/z-level-array.h>



/******************************************************************************
 ******************************************************************************
 *
 * CREATOR/DESTRUCTOR
 *
 ******************************************************************************
 *****************************************************************************/

/******************************************************************************
 *
 * picviz_z_level_array_new
 * WARNING !
 * the array is NOT initialized
 *
 *****************************************************************************/
picviz_z_level_array_t *picviz_z_level_array_new(int initial_row_count)
{
	picviz_z_level_array_t *zla;

	zla = (picviz_z_level_array_t *)malloc(sizeof(picviz_z_level_array_t));
	if (!zla) {
		PVLOG_ERROR("Cannot allocate z_level_array!\n");
		return NULL;
	}

	zla->array = (float *)calloc(PICVIZ_Z_LEVEL_ARRAY_MAX_SIZE, sizeof(float));
	if ( ! zla->array ) {
		PVLOG_ERROR("Cannot allocate array for z_level_array!\n");
		return NULL;
	}

	if ( (0<=initial_row_count) && (initial_row_count<=PICVIZ_Z_LEVEL_ARRAY_MAX_SIZE) ) {
		zla->row_count = initial_row_count;
	} else {
		PVLOG_ERROR("Cannot set row_count while creating picviz_z_level_array because it is out of range!\n");
	}

	return zla;
}



/******************************************************************************
 *
 * picviz_z_level_array_destroy
 *
 *****************************************************************************/
void picviz_z_level_array_destroy(picviz_z_level_array_t *zla)
{
	free(zla->array);
	free(zla);
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
 * picviz_z_level_array_get_row_count
 *
 *****************************************************************************/
int picviz_z_level_array_get_row_count(picviz_z_level_array_t *zla)
{
	return zla->row_count;
}


/******************************************************************************
 *
 * picviz_z_level_array_get_value
 *
 *****************************************************************************/
float picviz_z_level_array_get_value(picviz_z_level_array_t *zla, int row_index)
{
	return zla->array[row_index];
}



/******************************************************************************
 *
 * picviz_z_level_array_set_row_count
 *
 *****************************************************************************/
void picviz_z_level_array_set_row_count(picviz_z_level_array_t *zla, int new_row_count)
{
	if ( (0<=new_row_count) && (new_row_count<=PICVIZ_Z_LEVEL_ARRAY_MAX_SIZE) ) {
		zla->row_count = new_row_count;
	} else {
		PVLOG_ERROR("Cannot picviz_z_level_array_set_row_count because row_count is out of range!\n");
	}
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
 * picviz_z_level_array_initialize
 *
 *****************************************************************************/
// void picviz_z_level_array_initialize(picviz_z_level_array_t *zla)
// {
// 	int k;
// 
// 	for (k = 0; k < zla->row_count; k++) {
// 		zla->array[k] = 0.0f;
// 	}
// 
// }










