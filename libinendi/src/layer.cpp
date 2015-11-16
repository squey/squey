/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <stdio.h>
#include <stdlib.h>

#include <inendi/general.h>
#include <inendi/layer.h>

/**
 * \defgroup INENDILayers INENDI Layers
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
* inendi_layer_new
*
**********************************************************************/
inendi_layer_t *inendi_layer_new(const char *name)
{
	inendi_layer_t *layer;

	layer = (inendi_layer_t *)malloc(sizeof(inendi_layer_t));
	if ( ! layer ) {
		fprintf(stderr, "Cannot allocate a new layer!\n");
		return NULL;
	}

	layer->name = (char *)malloc(1024 * sizeof(char));

	strncpy(layer->name, name, 1000);

	layer->mode = INENDI_LAYER_NORMAL;
	layer->visible = 1;
	layer->locked = 0;
	layer->opacity = 1.0;
	layer->dynamic = 0;

	layer->selection = inendi_selection_new();
	layer->lines_properties = new Inendi::PVLinesProperties();

	return layer;
}


/**********************************************************************
*
* inendi_layer_destroy
*
**********************************************************************/
void inendi_layer_destroy(inendi_layer_t *layer)
{
	inendi_selection_destroy(layer->selection);
	free(layer->name);
	free(layer);
}

void inendi_layer_debug(inendi_layer_t *layer)
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
* inendi_layer_get_dynamic
*
**********************************************************************/
int inendi_layer_get_dynamic(inendi_layer_t *layer)
{
	return layer->dynamic;
}


/**********************************************************************
*
* inendi_layer_get_lines_properties
*
**********************************************************************/
Inendi::PVLinesProperties *inendi_layer_get_lines_properties(inendi_layer_t *layer)
{
	return layer->lines_properties;
}


/**********************************************************************
*
* inendi_layer_get_locked
*
**********************************************************************/
int inendi_layer_get_locked(inendi_layer_t *layer)
{
	return layer->locked;
}


/**********************************************************************
*
* inendi_layer_get_name
*
**********************************************************************/
char *inendi_layer_get_name(inendi_layer_t *layer)
{
	return layer->name;
}

/**********************************************************************
*
* inendi_layer_get_selection
*
**********************************************************************/
inendi_selection_t *inendi_layer_get_selection(inendi_layer_t *layer)
{
	return layer->selection;
}


/**********************************************************************
*
* inendi_layer_get_visible
*
**********************************************************************/
int inendi_layer_get_visible(inendi_layer_t *layer)
{
	return layer->visible;
}


/**********************************************************************
*
* inendi_layer_set_lines_properties_by_copy
*
**********************************************************************/
// void inendi_layer_set_lines_properties_by_copy(inendi_layer_t *layer, inendi_lines_properties_t *lp)
// {
// 	inendi_lines_properties_A2B_copy(lp, layer->lines_properties);
// }


/**********************************************************************
*
* inendi_layer_set_locked
*
**********************************************************************/
void inendi_layer_set_locked(inendi_layer_t *layer, int locked)
{
	layer->locked = locked;
}


/**********************************************************************
*
* inendi_layer_set_name
*
**********************************************************************/
void inendi_layer_set_name(inendi_layer_t *layer, char *name)
{
	strncpy(layer->name, name, 1000);
}


/**********************************************************************
*
* inendi_layer_set_selection_by_copy
*
**********************************************************************/
void inendi_layer_set_selection_by_copy(inendi_layer_t *layer, inendi_selection_t *selection)
{
	memcpy(layer->selection->table, selection->table, INENDI_SELECTION_NUMBER_OF_BYTES);
}


/**********************************************************************
*
* inendi_layer_set_visible
*
**********************************************************************/
void inendi_layer_set_visible(inendi_layer_t *layer, int visible)
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
 * inendi_layer_A2A_reset_to_empty_and_default_color
 *
 *****************************************************************************/
void inendi_layer_A2A_reset_to_empty_and_default_color(inendi_layer_t *a)
{
	inendi_selection_A2A_select_none(a->selection);
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
 * inendi_layer_A2B_copy
 *
 *****************************************************************************/
void inendi_layer_A2B_copy(inendi_layer_t *a, inendi_layer_t *b)
{
	b->lines_properties = a->lines_properties;
	// memcpy(b->lines_properties->table, a->lines_properties->table, INENDI_LINESPROPS_NUMBER_OF_BYTES);
	memcpy(b->selection->table, a->selection->table, INENDI_SELECTION_NUMBER_OF_BYTES);
}






/*@}*/
