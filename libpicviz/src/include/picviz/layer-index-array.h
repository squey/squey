/**
 * \file layer-index-array.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef _PICVIZ_LAYER_INDEX_ARRAY_H_
#define _PICVIZ_LAYER_INDEX_ARRAY_H_

#include <picviz/general.h>


#ifdef __cplusplus
 extern "C" {
#endif


#define PICVIZ_LAYER_INDEX_ARRAY_MAX_SIZE PICVIZ_LINES_MAX
/* WARNING! : should be the same as PICVIZ_LAYER_STACK_MAX_DEPTH
        but not used to avoid circular dependencies
*/
#define PICVIZ_LAYER_INDEX_ARRAY_MAX_VALUE 256


/******************************************************************************
 *
 * WARNING!
 *
 * It is important to get that the value in layer_index_array have the
 *  following meaning :
 *   0 : SPECIAL VALUE! : means that the line is not present in any layer
 *                        in the layer stack
 *   1-256 : means that the line appears first (upmost/higher value) at
 *           that given value.
 *
 * So be careful about the indexing when using arrays...
 *
 *****************************************************************************/

/******************************************************************************
 *
 * STRUCTURE
 *
 *****************************************************************************/

struct _picviz_layer_index_array_t {
	int *array;
	int row_count;
	int index_count;
};
typedef struct _picviz_layer_index_array_t picviz_layer_index_array_t;




/******************************************************************************
 ******************************************************************************
 *
 * CREATOR/DESTRUCTOR
 *
 ******************************************************************************
 *****************************************************************************/

LibPicvizDecl picviz_layer_index_array_t *picviz_layer_index_array_new(int initial_row_count);
LibPicvizDecl void picviz_layer_index_array_destroy(picviz_layer_index_array_t *lia);



/******************************************************************************
 ******************************************************************************
 *
 * GET/SET functions
 *
 ******************************************************************************
 *****************************************************************************/

LibPicvizDecl int picviz_layer_index_array_get_row_count(picviz_layer_index_array_t *lia);
LibPicvizDecl int picviz_layer_index_array_get_index_count(picviz_layer_index_array_t *lia);
LibPicvizDecl int picviz_layer_index_array_get_value(picviz_layer_index_array_t *lia, int row_index);



LibPicvizDecl void picviz_layer_index_array_set_row_count(picviz_layer_index_array_t *lia, int new_row_count);




/******************************************************************************
 ******************************************************************************
 *
 * SPECIFIC functions
 *
 ******************************************************************************
 *****************************************************************************/


LibPicvizDecl void picviz_layer_index_array_initialize(picviz_layer_index_array_t *layer_index_array);


#ifdef __cplusplus
 }
#endif

#endif /* _PICVIZ_LAYER_INDEX_ARRAY_H_ */
