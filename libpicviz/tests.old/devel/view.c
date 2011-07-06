#include <stdio.h>

#include <apr_tables.h>

#include <picviz/area.h>
#include <picviz/axis.h>
#include <picviz/axes-combination.h>
#include <picviz/view.h>
#include <picviz/layer.h>
#include <picviz/selection.h>
#include <picviz/line-properties.h>
#include <picviz/filter-library.h>
#include <picviz/utils.h>

#define LOGTYPE "csv"
#define LOGFILE "horrible.csv"
//#define LOGTYPE "syslog"
//#define LOGFILE "messages"
//#define LOGFILE "fake"


int main(int argc, char **argv)
{
	picviz_context_t *context;

	picviz_view_t *view;
	picviz_layer_t *layer;

	picviz_selection_t *selection;
	picviz_axes_combination_t *axescombination;
	picviz_axis_t *axis;
	picviz_filter_library_t *filterlib;

	picviz_lines_properties_t *linesprops;

	char *function_name;
	picviz_filter_function filter_function;

	apr_uint64_t count;

	context = picviz_init(argc, argv);

	view = picviz_h_view_create(context, LOGTYPE, LOGFILE);

	layer = picviz_view_layer_get(view, "default");

	/* picviz_selection_set_line(view->selection, 0, 1); */
	picviz_selection_select_all(view->selection);
	
	selection = picviz_selection_new();
	/* picviz_selection_set_line(selection, 0, 1); */
	/* picviz_selection_set_line(selection, 8, 1); */

	/* view->selection = picviz_selection_and(view->selection, selection); */
	/* printf("NOT INVERSE\n"); */
	count = picviz_view_total_lines_count(view);
	printf("count = %d\n", count);
	printf("number of selected lines:%llu\n", picviz_selection_count(view->selection, count));
	picviz_view_foreach_selected_lines_nraw(view, picviz_array_default_print_callback, NULL);

	for (count = 0; count < picviz_selection_count(view->selection, picviz_view_total_lines_count(view)); count++) {
		printf("count=%llu\n", count);
		/* printf("line id :%llu\n", picviz_selection_get_line_id_at_index(view->selection, count)); */
	}

	return 0;
	/* printf("first 5 lines states:%d,%d,%d,%d,%d\n", */
	/* 	picviz_selection_get_line(view->selection, 0), */
	/* 	picviz_selection_get_line(view->selection, 1), */
	/* 	picviz_selection_get_line(view->selection, 2), */
	/* 	picviz_selection_get_line(view->selection, 3), */
	/* 	picviz_selection_get_line(view->selection, 4)); */

	picviz_selection_inverse(view->selection);
	printf("INVERSED\n");
	printf("number of selected lines:%llu\n", picviz_selection_count(view->selection, count));
	picviz_view_foreach_selected_lines_nraw(view, picviz_array_default_print_callback, NULL);

	/* printf("first 5 lines states:%d,%d,%d,%d,%d\n", */
	/* 	picviz_selection_get_line(view->selection, 0), */
	/* 	picviz_selection_get_line(view->selection, 1), */
	/* 	picviz_selection_get_line(view->selection, 2), */
	/* 	picviz_selection_get_line(view->selection, 3), */
	/* 	picviz_selection_get_line(view->selection, 4)); */


	/* printf("GET SQUARE SELECTION\n"); */
	/* picviz_area_get_square_selection(view, 0.32, 1.234, 0.23, 4.32); */
	printf("layer name = %s\n", layer->name);

	picviz_line_set_color_r(layer->lines_properties, 0, 123);
	picviz_line_set_color_g(layer->lines_properties, 0, 111);
	picviz_line_set_color_b(layer->lines_properties, 0, 222);
	picviz_line_set_color_a(layer->lines_properties, 0, 255);

	printf("r=%d\n", picviz_line_get_color_r(layer->lines_properties, 0));
	printf("g=%d\n", picviz_line_get_color_g(layer->lines_properties, 0));
	printf("b=%d\n", picviz_line_get_color_b(layer->lines_properties, 0));
	printf("a=%d\n", picviz_line_get_color_a(layer->lines_properties, 0));

	axescombination = picviz_axes_combination_build_from_source(view->parent->parent->parent);
	axis = picviz_axes_combination_get_axis(axescombination, 1);

	picviz_terminate(context);

	filterlib = picviz_filter_library_new(context->pool);
	picviz_filter_library_append_function(filterlib, "less equal");
	function_name = picviz_filter_library_get_function_name_by_index(filterlib, 0);
	printf("Function at index 0:%s\n", function_name);


	linesprops = picviz_lines_properties_new();
	/* picviz_lines_properties_copy_to should be changed to picviz_lines_properties_A2B_copy if file still in use.... */
	picviz_lines_properties_copy_to(view->lines_properties, linesprops);
	/* filter_function = picviz_filter_library_get_function_ptr_by_index(filterlib, 0); */
	/* filter_function(NULL, NULL, NULL); */

	/* picviz_filtering_function_heatline(view, view->selection); */
	/* printf("number of functions:%d\n",picviz_filter_library_functions_name_count(filterlib)); */
	/* printf("number of types for function 0:%d\n",picviz_filter_library_function_types_for_index_count(filterlib, 0)); */
	/* printf("type 0:%s\n", picviz_filter_library_functions_types_get(filterlib,0,0)); */
	/* printf("type 1:%s\n", picviz_filter_library_functions_types_get(filterlib,0,1)); */
	/* printf("label 1:%s\n", picviz_filter_library_functions_label_get(filterlib,0,1)); */

	/* printf("View.selection -> LayerStack -> View.current_selection\n"); */
	/* picviz_view_process_layers_current_selection(view); */

	return 0;
}
