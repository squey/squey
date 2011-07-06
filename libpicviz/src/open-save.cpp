//! \file open-save.cpp
//! $Id: open-save.cpp 2489 2011-04-25 01:53:05Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <stdio.h>
#include <stdlib.h>

#include <pvcore/general.h>

#include <picviz/view.h>
#include <picviz/layer-stack.h>
#include <picviz/layer.h>

#include <picviz/open-save.h>

#define API_VERSION 1

/* 
 * PCV File Structure
 * 
 * Header
 * |PCV|API_VER|NUMBER_OF_LAYERS|SELECTED_LINES|LINE_PROPERTIES|
 * Data
 * |SIZE_LAYER_NAME|LAYER_NAME|SELECTED_LINES|LINE_PROPERTIES|
 */
int picviz_save(picviz_view_t *view, char *filename)
{
	FILE *fp;
	char api_ver = API_VERSION;
	int layers_number;
	int i;
	int size;

	picviz_layer_t *layer;


	fp = fopen(filename, "w");
	if (!fp) {
		PVLOG_ERROR("Cannot write to file '%s'\n", filename);
		return -1;
	}

	/* Header */
	fwrite("PCV|", 4, 1, fp);
	fwrite(&api_ver, sizeof(char), 1, fp);
	
	layers_number = picviz_layer_stack_get_layer_count(view->layer_stack);
	fwrite(&layers_number, sizeof(int), 1, fp);

	/* Data */
	for (i=0; i < layers_number; i++) {
		layer = picviz_layer_stack_get_layer_n(view->layer_stack, i);
		size = strlen(layer->name);
		fwrite(&size, sizeof(int), 1, fp);
		fwrite(layer->name, strlen(layer->name), 1, fp);
		fwrite(layer->selection->table, PICVIZ_SELECTION_NUMBER_OF_BYTES, 1, fp);
		fwrite(layer->lines_properties->table, PICVIZ_LINESPROPS_NUMBER_OF_BYTES, 1, fp);
	}

	fclose(fp);

	return 0;
}

static void picviz_open_read_layerstack(picviz_view_t *view, long offset, int layers_number, FILE *fp)
{
	int retval;
	int i;
	int size;
	char *str = NULL;
	picviz_layer_t *layer;

	if (view->layer_stack) picviz_layer_stack_destroy(view->layer_stack);

	view->layer_stack = picviz_layer_stack_new(NULL);
	view->layer_stack->parent = view;

	for (i=0; i < layers_number; i++) {
	  /* Size of layer name */
	  retval = fread(&size, sizeof(int), 1, fp);
	  str = (char *)malloc(size + 1);
	  retval = fread(str, size, 1, fp);
	  str[size] = '\0';

	  layer = picviz_layer_new(str);
	  free(str);

	  retval = fread(layer->selection->table, PICVIZ_SELECTION_NUMBER_OF_BYTES, 1, fp);
	  retval = fread(layer->lines_properties->table, PICVIZ_LINESPROPS_NUMBER_OF_BYTES, 1, fp);

	  /* fseek(fp, sizeof(picviz_selection_t) + sizeof(picviz_lines_properties_t), SEEK_CUR); */
	  /* retval = fread(layer->selection, sizeof(picviz_selection_t), 1, fp); */
	  /* retval = fread(layer->lines_properties, sizeof(picviz_lines_properties_t), 1, fp); */

	  picviz_layer_stack_append_layer(view->layer_stack, layer);
	  /* printf("layer->name='%s'\n", str); */
	}
  
}

int picviz_open_inline(picviz_view_t *view, char *filename)
{
	FILE *fp;
	int retval;

	long offset;

	char *str = NULL;
	picviz_layer_t *layer;
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
	picviz_open_read_layerstack(view, offset, layers_number, fp);
	/**********************************
	 * </LayerStack> 
	 **********************************/

	fclose(fp);

	return 0;
}

picviz_view_t *picviz_open(char *filename)
{
	picviz_view_t *view;

	view = picviz_view_new_basic();

	picviz_open_inline(view, filename);

	return view;
}

int picviz_open_is_picviz_type(char *filename)
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
