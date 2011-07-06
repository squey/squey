#include <pvcore/general.h>

#include <picviz/view.h>
#include <picviz/layer.h>
#include <picviz/filtering-function.h>
#include <picviz/selection.h>

void *picviz_filter_plugin_log_heatline(picviz_view_t *view, picviz_layer_t *input_layer, picviz_layer_t *output_layer, picviz_filtering_heatline_function function, float fmin, float fmax);
void picviz_filter_plugin_log_heatline_colorize_do(picviz_view_t *view, picviz_layer_t *input_layer, picviz_layer_t *output_layer, float ratio, PVRow line_id, float fmin, float fmax);
void picviz_filter_plugin_log_heatline_select_do(  picviz_view_t *view, picviz_layer_t *input_layer, picviz_layer_t *output_layer, float ratio, PVRow line_id, float fmin, float fmax);

