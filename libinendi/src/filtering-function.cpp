/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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

#include <inendi/general.h>
#include <inendi/color.h>
#include <inendi/selection.h>
#include <inendi/filtering-function.h>
#include <inendi/line-properties.h>
#include <inendi/limits.h>



/******************************************************************************
 *
 * inendi_filtering_function_compare
 *
 *****************************************************************************/
inendi_selection_t *inendi_filtering_function_compare(inendi_selection_t *selection, int axis_index, float plotted, inendi_comparison_t compare)
{

	switch(compare) {
	case INENDI_LESS:	/* < */
		printf("inendi less for %f\n", plotted);
		break;
	case INENDI_LESS_EQUAL:	/* <= */
		printf("inendi less equal for %f\n", plotted);
		break;
	case INENDI_EQUAL:	/* = */
		printf("inendi equal for %f\n", plotted);
		break;
	case INENDI_GREATER:	/* > */
		printf("inendi greater for %f\n", plotted);
		break;
	case INENDI_GREATER_EQUAL: /* >= */
		printf("inendi greater equal for %f\n", plotted);
		break;
	case INENDI_NOT:	/* ! */
		printf("inendi not for %f\n", plotted);
		break;
	default:
		fprintf(stderr, "Cannot compare, invalid operator '%d'\n", compare);
	}

	return selection;
}


/******************************************************************************
 *
 * inendi_filtering_function_less_equal
 *
 *****************************************************************************/
inendi_selection_t *inendi_filtering_function_less_equal(inendi_plotted_t *plotted, inendi_selection_t *selection, void *args)
{
	int axis_index;
	float plotval;

	axis_index = 0;
	plotval = 0.5;

	return inendi_filtering_function_compare(selection, axis_index, plotval, INENDI_EQUAL);
}

/******************************************************************************
 *
 * inendi_filtering_function_square_area
 *
 *****************************************************************************/
char *inendi_filtering_function_square_area(inendi_view_t *view, inendi_layer_t *input_layer, inendi_layer_t *output_layer, float xmin, float ymin, float xmax, float ymax)
{
	inendi_plotted_t *plotted;
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

	inendi_selection_A2A_select_none(output_layer->selection);
	inendi_lines_properties_A2B_copy(input_layer->lines_properties, output_layer->lines_properties);



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
					inendi_selection_set_line(output_layer->selection, line_index, 1);
					continue;
			}
			break;
	}

	
/*	for (axis_pos = axis_left; axis_pos <= axis_right - 1; axis_pos++) {
		x = axis_pos;
		xb = axis_pos + 1;
		x_left = inendi_max(xmin, x);
		x_right = inendi_min(xmax, xb);

		for ( line_index = 0; line_index < plotted->table->nelts; line_index++ ) {
			apr_array_header_t *plottedvalues = ((apr_array_header_t **)plotted->table->elts)[line_index];

			if (inendi_selection_get_line(input_layer->selection, line_index)) {

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
				inendi_selection_set_line(output_layer->selection, line_index, 1);
			}
		}
	}
*/

	return NULL;

}


/******************************************************************************
 *
 * inendi_filtering_function_square_area_slow
 *
 *****************************************************************************/
char *inendi_filtering_function_square_area_slow(inendi_view_t *view, inendi_layer_t *input_layer, inendi_layer_t *output_layer, float xmin, float ymin, float xmax, float ymax)
{
	inendi_plotted_t *plotted;
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

	inendi_selection_A2A_select_none(output_layer->selection);
	inendi_lines_properties_A2B_copy(input_layer->lines_properties, output_layer->lines_properties);

//  	printf("affichons plotted->table->nelts :%ull\n", plotted->table->nelts);
	for (axis_pos = axis_left; axis_pos <= axis_right - 1; axis_pos++) {
		x = axis_pos;
		xb = axis_pos + 1;
		x_left = inendi_max(xmin, x);
		x_right = inendi_min(xmax, xb);

		//--for ( line_index = 0; line_index < plotted->table->nelts; line_index++ ) {
		for ( line_index = 0; line_index < plotted->row_count; line_index++ ) {
			//--apr_array_header_t *plottedvalues = ((apr_array_header_t **)plotted->table->elts)[line_index];

			if (inendi_selection_get_line(input_layer->selection, line_index)) {

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
				inendi_selection_set_line(output_layer->selection, line_index, 1);
			}
		}
	}


	return NULL;

}

char *inendi_filtering_function_selectall(inendi_view_t *view, inendi_layer_t *input_layer, inendi_layer_t *output_layer)
{
	inendi_selection_A2A_select_all(output_layer->selection);

	return NULL;
}
