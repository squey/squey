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
#include <inendi/z-level-array.h>



/******************************************************************************
 ******************************************************************************
 *
 * CREATOR/DESTRUCTOR
 *
 ******************************************************************************
 *****************************************************************************/

/******************************************************************************
 *
 * inendi_z_level_array_new
 * WARNING !
 * the array is NOT initialized
 *
 *****************************************************************************/
inendi_z_level_array_t *inendi_z_level_array_new(int initial_row_count)
{
	inendi_z_level_array_t *zla;

	zla = (inendi_z_level_array_t *)malloc(sizeof(inendi_z_level_array_t));
	if (!zla) {
		PVLOG_ERROR("Cannot allocate z_level_array!\n");
		return NULL;
	}

	zla->array = (float *)calloc(INENDI_Z_LEVEL_ARRAY_MAX_SIZE, sizeof(float));
	if ( ! zla->array ) {
		PVLOG_ERROR("Cannot allocate array for z_level_array!\n");
		return NULL;
	}

	if ( (0<=initial_row_count) && (initial_row_count<=INENDI_Z_LEVEL_ARRAY_MAX_SIZE) ) {
		zla->row_count = initial_row_count;
	} else {
		PVLOG_ERROR("Cannot set row_count while creating inendi_z_level_array because it is out of range!\n");
	}

	return zla;
}



/******************************************************************************
 *
 * inendi_z_level_array_destroy
 *
 *****************************************************************************/
void inendi_z_level_array_destroy(inendi_z_level_array_t *zla)
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
 * inendi_z_level_array_get_row_count
 *
 *****************************************************************************/
int inendi_z_level_array_get_row_count(inendi_z_level_array_t *zla)
{
	return zla->row_count;
}


/******************************************************************************
 *
 * inendi_z_level_array_get_value
 *
 *****************************************************************************/
float inendi_z_level_array_get_value(inendi_z_level_array_t *zla, int row_index)
{
	return zla->array[row_index];
}



/******************************************************************************
 *
 * inendi_z_level_array_set_row_count
 *
 *****************************************************************************/
void inendi_z_level_array_set_row_count(inendi_z_level_array_t *zla, int new_row_count)
{
	if ( (0<=new_row_count) && (new_row_count<=INENDI_Z_LEVEL_ARRAY_MAX_SIZE) ) {
		zla->row_count = new_row_count;
	} else {
		PVLOG_ERROR("Cannot inendi_z_level_array_set_row_count because row_count is out of range!\n");
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
 * inendi_z_level_array_initialize
 *
 *****************************************************************************/
// void inendi_z_level_array_initialize(inendi_z_level_array_t *zla)
// {
// 	int k;
// 
// 	for (k = 0; k < zla->row_count; k++) {
// 		zla->array[k] = 0.0f;
// 	}
// 
// }










