//! \file z-level-array.h
//! $Id: z-level-array.h 2554 2011-05-03 19:40:46Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef _PICVIZ_Z_LEVEL_ARRAY_H_
#define _PICVIZ_Z_LEVEL_ARRAY_H_

#include <picviz/general.h>


#ifdef __cplusplus
 extern "C" {
#endif


#define PICVIZ_Z_LEVEL_ARRAY_MAX_SIZE PICVIZ_LINES_MAX



/******************************************************************************
 *
 * STRUCTURE
 *
 *****************************************************************************/

struct _picviz_z_level_array_t {
	float *array;
	int row_count;
};
typedef struct _picviz_z_level_array_t picviz_z_level_array_t;




/******************************************************************************
 ******************************************************************************
 *
 * CREATOR/DESTRUCTOR
 *
 ******************************************************************************
 *****************************************************************************/

LibPicvizDecl picviz_z_level_array_t *picviz_z_level_array_new(int initial_row_count);
LibPicvizDecl void picviz_z_level_array_destroy(picviz_z_level_array_t *zla);



/******************************************************************************
 ******************************************************************************
 *
 * GET/SET functions
 *
 ******************************************************************************
 *****************************************************************************/

LibPicvizDecl int picviz_z_level_array_get_row_count(picviz_z_level_array_t *zla);
LibPicvizDecl float picviz_z_level_array_get_value(picviz_z_level_array_t *zla, int row_index);

LibPicvizDecl void picviz_z_level_array_set_row_count(picviz_z_level_array_t *zla, int new_row_count);


/******************************************************************************
 ******************************************************************************
 *
 * SPECIFIC functions
 *
 ******************************************************************************
 *****************************************************************************/

// LibPicvizDecl void picviz_z_level_array_initialize(picviz_z_level_array_t *zla);


#ifdef __cplusplus
 }
#endif

#endif /* _PICVIZ_Z_LEVEL_ARRAY_H_ */
