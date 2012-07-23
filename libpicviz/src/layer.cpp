/**
 * \file layer.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <stdio.h>
#include <stdlib.h>

#include <picviz/general.h>
#include <picviz/layer.h>

/**
 * \defgroup PicvizLayers Picviz Layers
 * @{
 */

/******************************************************************************
 ******************************************************************************
 *
 * CREATOR/DESTRUCTOR
 *
 ******************************************************************************
 *****************************************************************************/

/**********************************************************************
*
* picviz_layer_new
*
**********************************************************************/
picviz_layer_t *picviz_layer_new(const char *name)
{
	picviz_layer_t *layer;

	layer = (picviz_layer_t *)malloc(sizeof(picviz_layer_t));
	if ( ! layer ) {
		fprintf(stderr, "Cannot allocate a new layer!\n");
		return NULL;
	}

	layer->name = (char *)malloc(1024 * sizeof(char));

	strncpy(layer->name, name, 1000);

	layer->mode = PICVIZ_LAYER_NORMAL;
	layer->visible = 1;
	layer->locked = 0;
	layer->opacity = 1.0;
	layer->dynamic = 0;

	layer->selection = picviz_selection_new();
	layer->lines_properties = new Picviz::PVLinesProperties();

	return layer;
}


/**********************************************************************
*
* picviz_layer_destroy
*
**********************************************************************/
void picviz_layer_destroy(picviz_layer_t *layer)
{
	picviz_selection_destroy(layer->selection);
	free(layer->name);
	free(layer);
}

void picviz_layer_debug(picviz_layer_t *layer)
{
	printf("Layer\n=====\n");
	printf("\tname:'%s'\n", layer->name);
	printf("\tmode:%d\n", layer->mode);
	printf("\tvisible:%d\n", layer->visible);
	printf("\tlocked:%d\n", layer->locked);
	printf("\tdynamic:%d\n", layer->dynamic);
	printf("\tindex:%d\n", layer->index);

}


/******************************************************************************
 ******************************************************************************
 *
 * GET/SET functions
 *
 ******************************************************************************
 *****************************************************************************/


/**********************************************************************
*
* picviz_layer_get_dynamic
*
**********************************************************************/
int picviz_layer_get_dynamic(picviz_layer_t *layer)
{
	return layer->dynamic;
}


/**********************************************************************
*
* picviz_layer_get_lines_properties
*
**********************************************************************/
Picviz::PVLinesProperties *picviz_layer_get_lines_properties(picviz_layer_t *layer)
{
	return layer->lines_properties;
}


/**********************************************************************
*
* picviz_layer_get_locked
*
**********************************************************************/
int picviz_layer_get_locked(picviz_layer_t *layer)
{
	return layer->locked;
}


/**********************************************************************
*
* picviz_layer_get_name
*
**********************************************************************/
char *picviz_layer_get_name(picviz_layer_t *layer)
{
	return layer->name;
}

/**********************************************************************
*
* picviz_layer_get_selection
*
**********************************************************************/
picviz_selection_t *picviz_layer_get_selection(picviz_layer_t *layer)
{
	return layer->selection;
}


/**********************************************************************
*
* picviz_layer_get_visible
*
**********************************************************************/
int picviz_layer_get_visible(picviz_layer_t *layer)
{
	return layer->visible;
}


/**********************************************************************
*
* picviz_layer_set_lines_properties_by_copy
*
**********************************************************************/
// void picviz_layer_set_lines_properties_by_copy(picviz_layer_t *layer, picviz_lines_properties_t *lp)
// {
// 	picviz_lines_properties_A2B_copy(lp, layer->lines_properties);
// }


/**********************************************************************
*
* picviz_layer_set_locked
*
**********************************************************************/
void picviz_layer_set_locked(picviz_layer_t *layer, int locked)
{
	layer->locked = locked;
}


/**********************************************************************
*
* picviz_layer_set_name
*
**********************************************************************/
void picviz_layer_set_name(picviz_layer_t *layer, char *name)
{
	strncpy(layer->name, name, 1000);
}


/**********************************************************************
*
* picviz_layer_set_selection_by_copy
*
**********************************************************************/
void picviz_layer_set_selection_by_copy(picviz_layer_t *layer, picviz_selection_t *selection)
{
	memcpy(layer->selection->table, selection->table, PICVIZ_SELECTION_NUMBER_OF_BYTES);
}


/**********************************************************************
*
* picviz_layer_set_visible
*
**********************************************************************/
void picviz_layer_set_visible(picviz_layer_t *layer, int visible)
{
	layer->visible = visible;
}



/******************************************************************************
 ******************************************************************************
 *
 * the set of operators of type A2A : inplace on A
 *
 ******************************************************************************
 *****************************************************************************/

/******************************************************************************
 *
 * picviz_layer_A2A_reset_to_empty_and_default_color
 *
 *****************************************************************************/
void picviz_layer_A2A_reset_to_empty_and_default_color(picviz_layer_t *a)
{
	picviz_selection_A2A_select_none(a->selection);
	a->lines_properties->reset_to_default_color();
}

/******************************************************************************
 ******************************************************************************
 *
 * the set of operators of type A2B : Operator(A, B) : A --> B
 *
 ******************************************************************************
 *****************************************************************************/

/******************************************************************************
 *
 * picviz_layer_A2B_copy
 *
 *****************************************************************************/
void picviz_layer_A2B_copy(picviz_layer_t *a, picviz_layer_t *b)
{
	b->lines_properties = a->lines_properties;
	// memcpy(b->lines_properties->table, a->lines_properties->table, PICVIZ_LINESPROPS_NUMBER_OF_BYTES);
	memcpy(b->selection->table, a->selection->table, PICVIZ_SELECTION_NUMBER_OF_BYTES);
}






/*@}*/
