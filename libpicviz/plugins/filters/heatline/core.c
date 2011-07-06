#include <apr_general.h>
#include <apr_strings.h>
#include <apr_tables.h>

#include <picviz/general.h>
#include <picviz/debug.h>
#include <picviz/view.h>
#include <picviz/layer.h>
#include <picviz/selection.h>
#include <picviz/line-properties.h>

#include <picviz/filtering-function.h>


/******************************************************************************
 *
 * picviz_filter_plugin_heatline_get_key
 *
 *****************************************************************************/
char *picviz_filter_plugin_heatline_get_key(picviz_view_t *view, apr_array_header_t *nrawvalues)
{
	apr_uint64_t column;
	char *key;

	key = "";

	for (column = 0; column < nrawvalues->nelts; column++) {
		const char *value = (const char *)((apr_array_header_t **)nrawvalues->elts)[column];
		picviz_source_t *source = picviz_view_get_source_parent(view);

		if (picviz_format_is_key_axis(source->format, column+1)) {
			key = apr_pstrcat(view->pool, key, value, NULL);
		}
	}

	return key;
}


/******************************************************************************
 *
 * picviz_filter_plugin_heatline_colorize_do
 *
 *****************************************************************************/
void picviz_filter_plugin_heatline_colorize_do(picviz_view_t *view, picviz_layer_t *input_layer, picviz_layer_t *output_layer, float ratio, apr_uint64_t line_id, float fmin, float fmax)
{

	picviz_color_t *color;

	color = picviz_color_new();
	picviz_color_set_hsv(color, 120*(1 - ratio), 1.0, 1.0);

	picviz_lines_properties_line_set_rgb_from_color(output_layer->lines_properties, line_id, color);

	/* The color structure can be safely destroyed because it's arguments have been fully copied */
	picviz_color_destroy(color);
}




/******************************************************************************
 *
 * picviz_filter_plugin_heatline_select_do
 *
 *****************************************************************************/
void picviz_filter_plugin_heatline_select_do(picviz_view_t *view, picviz_layer_t *input_layer, picviz_layer_t *output_layer, float ratio, apr_uint64_t line_id, float fmin, float fmax) {

	if ((ratio > fmax) || (ratio < fmin)) {
		picviz_selection_set_line(output_layer->selection, line_id, 0);
	}
}



/******************************************************************************
 *
 * picviz_filter_plugin_heatline
 *
 *****************************************************************************/
char *picviz_filter_plugin_heatline(picviz_view_t *view, picviz_layer_t *input_layer, picviz_layer_t *output_layer, picviz_filtering_heatline_function function, float fmin, float fmax)
{
	apr_uint64_t nb_lines;
	apr_uint64_t counter;
	apr_uint64_t *count_frequency;
	apr_uint64_t count_d;
	apr_uint64_t highest_frequency;
	int columns;

	apr_hash_t *lines_hash;

	picviz_nraw_t *nraw;

	char *key;

	float ratio;


	/* picviz_debug(PICVIZ_DEBUG_CRITICAL, "WE ARE ABSOLUTELY NOT IN LOG FREQUENCY\n", ratio); */


	highest_frequency = 1;

	lines_hash = apr_hash_make(view->pool);

	picviz_selection_A2B_copy(input_layer->selection, output_layer->selection);
	
	nb_lines = picviz_array_get_elts_count(view->parent->table);
	/* 1st round: we calculate all the frequencies */
	for (counter = 0; counter < nb_lines; counter++) {
		apr_array_header_t *nrawvalues;
		nraw = picviz_view_get_nraw_parent(view);

		nrawvalues = ((apr_array_header_t **)nraw->nraws->elts)[counter];
		key = picviz_filter_plugin_heatline_get_key(view, nrawvalues);

		count_frequency = apr_hash_get(lines_hash, key, strlen(key));
		if (!count_frequency) {
			count_frequency = (int *)1;
			apr_hash_set(lines_hash, key, strlen(key), count_frequency);
		} else {
			count_d = (int *)count_frequency;
			count_d++;
			count_frequency = (int *)count_d;
			if (count_d > highest_frequency) {
				highest_frequency = count_d;
			}
			apr_hash_set(lines_hash, key, strlen(key), count_frequency);
		}
	}

	/* 2nd round: we get the color from the ratio compared with the key and the frequency */
	for (counter = 0; counter < nb_lines; counter++) {
		apr_array_header_t *nrawvalues;
		nraw = picviz_view_get_nraw_parent(view);

		nrawvalues = ((apr_array_header_t **)nraw->nraws->elts)[counter];
		key = picviz_filter_plugin_heatline_get_key(view, nrawvalues);

		count_frequency = apr_hash_get(lines_hash, key, strlen(key));

		count_d = (int *)count_frequency;
		ratio = (float)count_d / highest_frequency;

		function(view, input_layer, output_layer, ratio, counter, fmin, fmax);
	}

// 	return selection;
	return NULL;
}

