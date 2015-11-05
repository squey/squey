/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <stdio.h>
#include <stdlib.h>

#include <pvkernel/core/general.h>

#include <inendi/view.h>
#include <inendi/layer-stack.h>
#include <inendi/layer.h>

#include <inendi/open-save.h>

#define API_VERSION 1

/* 
 * PCV File Structure
 * 
 * Header
 * |PCV|API_VER|NUMBER_OF_LAYERS|SELECTED_LINES|LINE_PROPERTIES|
 * Data
 * |SIZE_LAYER_NAME|LAYER_NAME|SELECTED_LINES|LINE_PROPERTIES|
 */
int inendi_save(inendi_view_t *view, char *filename)
{
	FILE *fp;
	char api_ver = API_VERSION;
	int layers_number;
	int i;
	int size;

	inendi_layer_t *layer;


	fp = fopen(filename, "w");
	if (!fp) {
		PVLOG_ERROR("Cannot write to file '%s'\n", filename);
		return -1;
	}

	/* Header */
	fwrite("PCV|", 4, 1, fp);
	fwrite(&api_ver, sizeof(char), 1, fp);
	
	layers_number = inendi_layer_stack_get_layer_count(view->layer_stack);
	fwrite(&layers_number, sizeof(int), 1, fp);

	/* Data */
	for (i=0; i < layers_number; i++) {
		layer = inendi_layer_stack_get_layer_n(view->layer_stack, i);
		size = strlen(layer->name);
		fwrite(&size, sizeof(int), 1, fp);
		fwrite(layer->name, strlen(layer->name), 1, fp);
		fwrite(layer->selection->table, INENDI_SELECTION_NUMBER_OF_BYTES, 1, fp);
		fwrite(layer->lines_properties->table, INENDI_LINESPROPS_NUMBER_OF_BYTES, 1, fp);
	}

	fclose(fp);

	return 0;
}

static void inendi_open_read_layerstack(inendi_view_t *view, long offset, int layers_number, FILE *fp)
{
	int retval;
	int i;
	int size;
	char *str = NULL;
	inendi_layer_t *layer;

	if (view->layer_stack) inendi_layer_stack_destroy(view->layer_stack);

	view->layer_stack = inendi_layer_stack_new(NULL);
	view->layer_stack->parent = view;

	for (i=0; i < layers_number; i++) {
	  /* Size of layer name */
	  retval = fread(&size, sizeof(int), 1, fp);
	  str = (char *)malloc(size + 1);
	  retval = fread(str, size, 1, fp);
	  str[size] = '\0';

	  layer = inendi_layer_new(str);
	  free(str);

	  retval = fread(layer->selection->table, INENDI_SELECTION_NUMBER_OF_BYTES, 1, fp);
	  retval = fread(layer->lines_properties->table, INENDI_LINESPROPS_NUMBER_OF_BYTES, 1, fp);

	  /* fseek(fp, sizeof(inendi_selection_t) + sizeof(inendi_lines_properties_t), SEEK_CUR); */
	  /* retval = fread(layer->selection, sizeof(inendi_selection_t), 1, fp); */
	  /* retval = fread(layer->lines_properties, sizeof(inendi_lines_properties_t), 1, fp); */

	  inendi_layer_stack_append_layer(view->layer_stack, layer);
	  /* printf("layer->name='%s'\n", str); */
	}
  
}

int inendi_open_inline(inendi_view_t *view, char *filename)
{
	FILE *fp;
	int retval;

	long offset;

	char *str = NULL;
	inendi_layer_t *layer;
	int layers_number;

	fp = fopen(filename, "r");
	if (!fp) {
		PVLOG_ERROR("Cannot open file '%s'\n", filename);
		return -1;
	}

	offset = 5L;
	fseek(fp, offset, SEEK_SET);

	retval = fread(&layers_number, sizeof(int), 1, fp);
	PVLOG_DEBUG("We have %d layer(s)\n", layers_number);

	/**********************************
	 * <LayerStack> 
	 **********************************/
	inendi_open_read_layerstack(view, offset, layers_number, fp);
	/**********************************
	 * </LayerStack> 
	 **********************************/

	fclose(fp);

	return 0;
}

inendi_view_t *inendi_open(char *filename)
{
	inendi_view_t *view;

	view = inendi_view_new_basic();

	inendi_open_inline(view, filename);

	return view;
}

int inendi_open_is_inendi_type(char *filename)
{
	FILE *fp;

	char pcv[4];

	int retval;


	fp = fopen(filename, "r");
	if (!fp) {
		PVLOG_ERROR("Cannot open file '%s'\n", filename);
		return -1;
	}

	fseek(fp, 0, SEEK_SET);
	retval = fread(&pcv, 4, 1, fp);
	pcv[4] = '\0';

	retval = strcmp(pcv, "PCV|");

	fclose(fp);

	if (!retval) return 1;

	return 0;
}
