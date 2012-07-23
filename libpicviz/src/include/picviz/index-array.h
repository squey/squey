/**
 * \file index-array.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef _PICVIZ_INDEX_ARRAY_H_
#define _PICVIZ_INDEX_ARRAY_H_

#include <picviz/general.h>


#ifdef __cplusplus
 extern "C" {
#endif


#define PICVIZ_INDEX_ARRAY_MAX_SIZE PICVIZ_LINES_MAX


/******************************************************************************
 *
 * STRUCTURE
 *
 *****************************************************************************/

struct _picviz_index_array_t {
	int *array;
	int row_count;
	int index_count;
};
typedef struct _picviz_index_array_t picviz_index_array_t;




/******************************************************************************
 ******************************************************************************
 *
 * CREATOR/DESTRUCTOR
 *
 ******************************************************************************
 *****************************************************************************/

LibPicvizDecl picviz_index_array_t *picviz_index_array_new(int initial_row_count);
LibPicvizDecl void picviz_index_array_destroy(picviz_index_array_t *ia);



/******************************************************************************
 ******************************************************************************
 *
 * GET/SET functions
 *
 ******************************************************************************
 *****************************************************************************/

// LibPicvizDecl int picviz_index_array_get_row_count(picviz_index_array_t *ia);
LibPicvizDecl int picviz_index_array_get_index_count(picviz_index_array_t *ia);


// LibPicvizDecl void picviz_index_array_set_row_count(picviz_index_array_t *ia, int new_row_count);
LibPicvizDecl void picviz_index_array_set_from_selection(picviz_index_array_t *ia, picviz_selection_t *selection);



/******************************************************************************
 ******************************************************************************
 *
 * SPECIFIC functions
 *
 ******************************************************************************
 *****************************************************************************/


// LibPicvizDecl void picviz_index_array_reset_linear(picviz_index_array_t *index_array);


#ifdef __cplusplus
 }
#endif

#endif /* _PICVIZ_INDEX_ARRAY_H_ */
