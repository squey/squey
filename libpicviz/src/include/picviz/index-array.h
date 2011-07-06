//! \file index-array.h
//! $Id: index-array.h 2548 2011-05-03 15:46:52Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

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

LibExport picviz_index_array_t *picviz_index_array_new(int initial_row_count);
LibExport void picviz_index_array_destroy(picviz_index_array_t *ia);



/******************************************************************************
 ******************************************************************************
 *
 * GET/SET functions
 *
 ******************************************************************************
 *****************************************************************************/

// LibExport int picviz_index_array_get_row_count(picviz_index_array_t *ia);
LibExport int picviz_index_array_get_index_count(picviz_index_array_t *ia);


// LibExport void picviz_index_array_set_row_count(picviz_index_array_t *ia, int new_row_count);
LibExport void picviz_index_array_set_from_selection(picviz_index_array_t *ia, picviz_selection_t *selection);



/******************************************************************************
 ******************************************************************************
 *
 * SPECIFIC functions
 *
 ******************************************************************************
 *****************************************************************************/


// LibExport void picviz_index_array_reset_linear(picviz_index_array_t *index_array);


#ifdef __cplusplus
 }
#endif

#endif /* _PICVIZ_INDEX_ARRAY_H_ */
