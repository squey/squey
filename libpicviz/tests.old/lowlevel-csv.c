#include <stdio.h>

#include <picviz/general.h>
#include <picviz/datatreerootitem.h>
#include <picviz/scene.h>
#include <picviz/source.h>
#include <picviz/mapping.h>
#include <picviz/mapped.h>
#include <picviz/plotted.h>
#include <picviz/view.h>
#include <picviz/layer.h>
#include <picviz/selection.h>

#define LOGTYPE "csv"
#define LOGFILE "nmap.csv"

int main(int argc, char **argv)
{
	picviz_datatreerootitem_t *datatree;
	picviz_scene_t *scene;
	picviz_source_t *source;
	picviz_mapping_t *mapping;
	picviz_mapped_t *mapped;
	picviz_plotting_t *plotting;
	picviz_plotted_t *plotted;
	picviz_view_t *view;
	picviz_layer_t *layer;
	picviz_selection_t *selection;

	int retval;

#include "test-env.h"

	picviz_init(argc, NULL);

	datatree = picviz_datatreerootitem_new();
	scene = picviz_scene_new(datatree, "default");

	source = picviz_source_new(scene);
	picviz_source_file_append(source, LOGTYPE, LOGFILE, NULL);
	mapping = picviz_mapping_new(source);
	mapped = picviz_mapped_make(mapping);
	plotting = picviz_plotting_new(mapped);
	plotted = picviz_plotted_build(plotting);
	retval = picviz_plotted_csv_export(plotted, "lowlevel-plotted.csv");

	view = picviz_view_new(plotted);
	layer = picviz_layer_new("default");
	//picviz_view_layer_append(view, layer);

	selection = picviz_selection_new();
	picviz_selection_A2A_select_all(selection);
	picviz_layer_set_selection_by_copy(layer, selection);

	printf("Success\n");

	picviz_terminate();

	return 0;
}
