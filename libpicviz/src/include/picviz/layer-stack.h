//! \file layer-stack.h
//! $Id: layer-stack.h 2569 2011-05-04 06:15:53Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef _PICVIZ_LAYER_STACK_H_
#define _PICVIZ_LAYER_STACK_H_

#include <picviz/general.h>
#include <picviz/layer.h>
#include <picviz/layer-index-array.h>
#include <picviz/selection.h>


#define PICVIZ_LAYER_STACK_MAX_DEPTH 256

#ifdef __cplusplus
 extern "C" {
#endif


/******************************************************************************
 *
 * STRUCTURE
 *
 *****************************************************************************/

struct _picviz_layer_stack_t {
	void *parent;

	picviz_layer_index_array_t *lia;
	int layer_count; // layer_count < 256
	int selected_layer_index;
	picviz_layer_t *table[PICVIZ_LAYER_STACK_MAX_DEPTH];

};
typedef struct _picviz_layer_stack_t picviz_layer_stack_t;


/******************************************************************************
 ******************************************************************************
 *
 * CREATOR/DESTRUCTOR
 *
 ******************************************************************************
 *****************************************************************************/

LibExport picviz_layer_stack_t *picviz_layer_stack_new(PVRow row_count);
LibExport void picviz_layer_stack_destroy(picviz_layer_stack_t *layer_stack);




/******************************************************************************
 ******************************************************************************
 *
 * GET/SET functions
 *
 ******************************************************************************
 *****************************************************************************/

LibExport int picviz_layer_stack_get_layer_count(picviz_layer_stack_t *layer_stack);
LibExport picviz_layer_t *picviz_layer_stack_get_layer_n(picviz_layer_stack_t *layer_stack, int n);
LibExport picviz_layer_t *picviz_layer_stack_get_selected_layer(picviz_layer_stack_t *layer_stack);
LibExport int picviz_layer_stack_get_selected_layer_index(picviz_layer_stack_t *layer_stack);

LibExport void picviz_layer_stack_set_selected_layer_index(picviz_layer_stack_t *layer_stack, int index);


/******************************************************************************
 ******************************************************************************
 *
 * GENERIC FUNCTIONS ON THE LAYER STACK
 *
 ******************************************************************************
 *****************************************************************************/


LibExport void picviz_layer_stack_process(picviz_layer_stack_t *layer_stack, picviz_layer_t *output_layer, PVRow row_count);
LibExport void picviz_layer_stack_update_layer_index_array_completely(picviz_layer_stack_t *layer_stack);


/******************************************************************************
 ******************************************************************************
 *
 * FUNCTIONS TO MANIPULATE THE LAYERS IN THE STACK
 *
 ******************************************************************************
 *****************************************************************************/

LibExport void picviz_layer_stack_append_layer(picviz_layer_stack_t *layer_stack, picviz_layer_t *layer);
LibExport void picviz_layer_stack_append_new_layer(picviz_layer_stack_t *layer_stack);
// LibExport void picviz_layer_stack_append_new_layer_from_layer(picviz_layer_stack_t *layer_stack, picviz_layer_t *layer);
LibExport void picviz_layer_stack_append_new_layer_from_selection_and_lines_properties(picviz_layer_stack_t *layer_stack, picviz_selection_t *selection, Picviz::PVLinesProperties *lines_properties);

LibExport void picviz_layer_stack_layer_delete_by_index(picviz_layer_stack_t *layer_stack, int index);
LibExport void picviz_layer_stack_delete_selected_layer(picviz_layer_stack_t *layer_stack);

LibExport picviz_layer_t *picviz_layer_stack_layer_get_by_index(picviz_layer_stack_t *layer_stack, int index);
// LibExport picviz_layer_t *picviz_layer_stack_layer_get_by_name(picviz_layer_stack_t *layer_stack, char *name);

LibExport int picviz_layer_stack_move_layer_down(picviz_layer_stack_t *layer_stack, int index);
LibExport int picviz_layer_stack_move_layer_up(picviz_layer_stack_t *layer_stack, int index);
LibExport void picviz_layer_stack_move_selected_layer_down(picviz_layer_stack_t *layer_stack);
LibExport void picviz_layer_stack_move_selected_layer_up(picviz_layer_stack_t *layer_stack);

LibExport void picviz_layer_stack_debug(picviz_layer_stack_t *layer_stack);





#ifdef __cplusplus
 }
#endif

#endif /* _PICVIZ_LAYER_STACK_H_ */
