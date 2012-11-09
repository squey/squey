#ifndef __PICVIZ_COMMON_H__
#define __PICVIZ_COMMON_H__

namespace Picviz {
namespace common {
	void load_filters();

	// Plugins loading
	int load_layer_filters();
	int load_mapping_filters();
	int load_plotting_filters();
	int load_row_filters();
	int load_axis_computation_filters();
	int load_sorting_functions_filters();
}
}

#endif
