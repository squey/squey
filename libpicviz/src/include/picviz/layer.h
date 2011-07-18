//! \file layer.h
//! $Id: layer.h 2541 2011-05-03 13:03:10Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef _PICVIZ_LAYER_H_
#define _PICVIZ_LAYER_H_

#include <picviz/general.h>
#include <picviz/selection.h>

#include <picviz/PVLinesProperties.h>

#ifdef __cplusplus
 extern "C" {
#endif


#define PICVIZ_LAYER_NAME_MAXLEN 1000

/******************************************************************************
 *
 * ENUM
 *
 *****************************************************************************/

enum _picviz_layer_mode_t {
	PICVIZ_LAYER_NORMAL,
	PICVIZ_LAYER_DIFFERENCE,
	PICVIZ_LAYER_ADDITION,
	PICVIZ_LAYER_SUBSTRACT,	
};
typedef enum _picviz_layer_mode_t picviz_layer_mode_t;



/******************************************************************************
 *
 * STRUCTURE
 *
 *****************************************************************************/

struct _picviz_layer_t {
	char *name;
	picviz_layer_mode_t mode;
	picviz_selection_t *selection;
	Picviz::PVLinesProperties *lines_properties;
	int visible;
	int locked;
	float opacity;
	int dynamic;
	uint16_t index;
};
typedef struct _picviz_layer_t picviz_layer_t;



/******************************************************************************
 ******************************************************************************
 *
 * CREATOR/DESTRUCTOR
 *
 ******************************************************************************
 *****************************************************************************/

LibPicvizDecl picviz_layer_t *picviz_layer_new(const char *name);
LibPicvizDecl void picviz_layer_destroy(picviz_layer_t *layer);




/******************************************************************************
 ******************************************************************************
 *
 * GET/SET functions
 *
 ******************************************************************************
 *****************************************************************************/

LibPicvizDecl int picviz_layer_get_dynamic(picviz_layer_t *layer);
LibPicvizDecl int picviz_layer_get_locked(picviz_layer_t *layer);
LibPicvizDecl char *picviz_layer_get_name(picviz_layer_t *layer);
LibPicvizDecl picviz_selection_t *picviz_layer_get_selection(picviz_layer_t *layer);
LibPicvizDecl Picviz::PVLinesProperties *picviz_layer_get_lines_properties(picviz_layer_t *layer);
LibPicvizDecl int picviz_layer_get_visible(picviz_layer_t *layer);

/* LibPicvizDecl void picviz_layer_set_lines_properties_by_copy(picviz_layer_t *layer, picviz_lines_properties_t *lp); */
LibPicvizDecl void picviz_layer_set_locked(picviz_layer_t *layer, int locked);
LibPicvizDecl void picviz_layer_set_name(picviz_layer_t *layer, char *name);
LibPicvizDecl void picviz_layer_set_selection_by_copy(picviz_layer_t *layer, picviz_selection_t *selection);
LibPicvizDecl void picviz_layer_set_visible(picviz_layer_t *layer, int visible);




/******************************************************************************
 ******************************************************************************
 *
 * the set of operators of type A2A : inplace on A
 *
 ******************************************************************************
 *****************************************************************************/

LibPicvizDecl void picviz_layer_A2A_reset_to_empty_and_default_color(picviz_layer_t *a);



/******************************************************************************
 ******************************************************************************
 *
 * the set of operators of type A2B : Operator(A, B) : A --> B
 *
 ******************************************************************************
 *****************************************************************************/

LibPicvizDecl void picviz_layer_A2B_copy(picviz_layer_t *a, picviz_layer_t *b);

LibPicvizDecl void picviz_layer_debug(picviz_layer_t *layer);





#ifdef __cplusplus
 }
#endif

#endif /* _PICVIZ_LAYER_H_ */
