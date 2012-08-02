/**
 * \file filtering-function.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <apr_general.h>
#include <apr_pools.h>
#include <apr_strings.h>
#include <apr_hash.h>

#if TIME_WITH_SYS_TIME
# include <sys/time.h>
# include <time.h>
#else
# if HAVE_SYS_TIME_H
#  include <sys/time.h>
# else
#  include <time.h>
# endif
#endif


//#include <win32-strptime.h>

#include <picviz/general.h>
#include <picviz/color.h>
#include <picviz/selection.h>
#include <picviz/filtering-function.h>
#include <picviz/line-properties.h>
#include <picviz/limits.h>



/******************************************************************************
 *
 * picviz_filtering_function_compare
 *
 *****************************************************************************/
picviz_selection_t *picviz_filtering_function_compare(picviz_selection_t *selection, int axis_index, float plotted, picviz_comparison_t compare)
{

	switch(compare) {
	case PICVIZ_LESS:	/* < */
		printf("picviz less for %f\n", plotted);
		break;
	case PICVIZ_LESS_EQUAL:	/* <= */
		printf("picviz less equal for %f\n", plotted);
		break;
	case PICVIZ_EQUAL:	/* = */
		printf("picviz equal for %f\n", plotted);
		break;
	case PICVIZ_GREATER:	/* > */
		printf("picviz greater for %f\n", plotted);
		break;
	case PICVIZ_GREATER_EQUAL: /* >= */
		printf("picviz greater equal for %f\n", plotted);
		break;
	case PICVIZ_NOT:	/* ! */
		printf("picviz not for %f\n", plotted);
		break;
	default:
		fprintf(stderr, "Cannot compare, invalid operator '%d'\n", compare);
	}

	return selection;
}


/******************************************************************************
 *
 * picviz_filtering_function_less_equal
 *
 *****************************************************************************/
picviz_selection_t *picviz_filtering_function_less_equal(picviz_plotted_t *plotted, picviz_selection_t *selection, void *args)
{
	int axis_index;
	float plotval;

	axis_index = 0;
	plotval = 0.5;

	return picviz_filtering_function_compare(selection, axis_index, plotval, PICVIZ_EQUAL);
}

/******************************************************************************
 *
 * picviz_filtering_function_square_area
 *
 *****************************************************************************/
char *picviz_filtering_function_square_area(picviz_view_t *view, picviz_layer_t *input_layer, picviz_layer_t *output_layer, float xmin, float ymin, float xmax, float ymax)
{
	picviz_plotted_t *plotted;
	apr_uint64_t line_index;
	apr_array_header_t *plottedvalues;
	
	float axis_left;
	float axis_right;
	float axis_pos;
	float inner_absciss_max;
	float inner_absciss_min;
	float inner_absciss_second;
	float k;
	int delta_inner_absciss;
	float ans1, ans2;
	float x, y, xb, yb;
	float x_left, x_right;
	int res_count;

	plotted = view->parent;
	axis_left = floorf(xmin);
	inner_absciss_min = axis_left + 1;
	axis_right = floorf(xmax + 1);
	inner_absciss_max = axis_right - 1;
	delta_inner_absciss = (int)(inner_absciss_max - inner_absciss_min);

	picviz_selection_A2A_select_none(output_layer->selection);
	picviz_lines_properties_A2B_copy(input_layer->lines_properties, output_layer->lines_properties);



	switch (delta_inner_absciss) {
		case (-1):

			break;

		case (0):

			break;

		default:
			inner_absciss_second = inner_absciss_min + 1;
			//--for ( line_index = 0; line_index < plotted->table->nelts; line_index++ ) {
			for ( line_index = 0; line_index < plotted->row_count; line_index++ ) {
				//--plottedvalues = ((apr_array_header_t **)plotted->table->elts)[line_index];
				//--if ( ((float *)plottedvalues->elts)[(int)inner_absciss_min] < ymin ) {
				if ( (plotted->array)[line_index * plotted->column_count + (int)inner_absciss_min] < ymin ) {
					for ( k=inner_absciss_second; k <= inner_absciss_max; k++) {
						//--if ( ((float *)plottedvalues->elts)[(int)k] >= ymin ) {
						if ( (plotted->array)[line_index * plotted->column_count + (int)k] >= ymin ) {
							goto set_line_select;
						}
					}
				//--} else if ( ((float *)plottedvalues->elts)[(int)inner_absciss_min] > ymax) {
				} else if ( (plotted->array)[line_index * plotted->column_count + (int)inner_absciss_min] > ymax) {
					for ( k=inner_absciss_second; k <= inner_absciss_max; k++) {
						//--if ( ((float *)plottedvalues->elts)[(int)k] <= ymax ) {
						if ( (plotted->array)[line_index * plotted->column_count + (int)k] <= ymax ) {
							goto set_line_select;
						}
					}
				} else {
					goto set_line_select;
				}
				continue;
				set_line_select:
					picviz_selection_set_line(output_layer->selection, line_index, 1);
					continue;
			}
			break;
	}

	
/*	for (axis_pos = axis_left; axis_pos <= axis_right - 1; axis_pos++) {
		x = axis_pos;
		xb = axis_pos + 1;
		x_left = picviz_max(xmin, x);
		x_right = picviz_min(xmax, xb);

		for ( line_index = 0; line_index < plotted->table->nelts; line_index++ ) {
			apr_array_header_t *plottedvalues = ((apr_array_header_t **)plotted->table->elts)[line_index];

			if (picviz_selection_get_line(input_layer->selection, line_index)) {

				res_count = 0;
				y = ((float *)plottedvalues->elts)[(int)axis_pos];
				yb = ((float *)plottedvalues->elts)[(int)axis_pos+1];

				ans1 = ((xb - x)*(ymax - y) - (yb - y)*(x_right - x)) * ((xb - x)*(ymin - y) - (yb - y)*(x_left - x));
				ans2 = ((xb - x)*(ymin - y) - (yb - y)*(x_right - x)) * ((xb - x)*(ymax - y) - (yb - y)*(x_left - x));

				if (ans1 <= 0) {
					res_count++;
				} else if (ans2 <= 0) {
					res_count++;
				}

			}

			if (res_count) {
				picviz_selection_set_line(output_layer->selection, line_index, 1);
			}
		}
	}
*/

	return NULL;

}


/******************************************************************************
 *
 * picviz_filtering_function_square_area_slow
 *
 *****************************************************************************/
char *picviz_filtering_function_square_area_slow(picviz_view_t *view, picviz_layer_t *input_layer, picviz_layer_t *output_layer, float xmin, float ymin, float xmax, float ymax)
{
	picviz_plotted_t *plotted;
	apr_uint64_t line_index;

	float axis_left;
	float axis_right;
	float axis_pos;
	float ans1, ans2;
	float x, y, xb, yb;
	float x_left, x_right;
	int res_count;

	plotted = view->parent;
	axis_left = floorf(xmin);
	axis_right = floorf(xmax + 1);

	picviz_selection_A2A_select_none(output_layer->selection);
	picviz_lines_properties_A2B_copy(input_layer->lines_properties, output_layer->lines_properties);

//  	printf("affichons plotted->table->nelts :%ull\n", plotted->table->nelts);
	for (axis_pos = axis_left; axis_pos <= axis_right - 1; axis_pos++) {
		x = axis_pos;
		xb = axis_pos + 1;
		x_left = picviz_max(xmin, x);
		x_right = picviz_min(xmax, xb);

		//--for ( line_index = 0; line_index < plotted->table->nelts; line_index++ ) {
		for ( line_index = 0; line_index < plotted->row_count; line_index++ ) {
			//--apr_array_header_t *plottedvalues = ((apr_array_header_t **)plotted->table->elts)[line_index];

			if (picviz_selection_get_line(input_layer->selection, line_index)) {

				res_count = 0;
				//--y = ((float *)plottedvalues->elts)[(int)axis_pos];
				y = (plotted->array)[line_index * plotted->column_count + (int)axis_pos];
				//--yb = ((float *)plottedvalues->elts)[(int)axis_pos+1];
				yb = (plotted->array)[line_index * plotted->column_count + (int)axis_pos + 1];

				ans1 = ((xb - x)*(ymax - y) - (yb - y)*(x_right - x)) * ((xb - x)*(ymin - y) - (yb - y)*(x_left - x));
				ans2 = ((xb - x)*(ymin - y) - (yb - y)*(x_right - x)) * ((xb - x)*(ymax - y) - (yb - y)*(x_left - x));

				if (ans1 <= 0) {
					res_count++;
				} else if (ans2 <= 0) {
					res_count++;
				}

			}

			if (res_count) {
				picviz_selection_set_line(output_layer->selection, line_index, 1);
			}
		}
	}


	return NULL;

}

char *picviz_filtering_function_selectall(picviz_view_t *view, picviz_layer_t *input_layer, picviz_layer_t *output_layer)
{
	picviz_selection_A2A_select_all(output_layer->selection);

	return NULL;
}
