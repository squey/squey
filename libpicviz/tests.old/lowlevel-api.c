#include <stdio.h>

#include <picviz/general.h>
#include <picviz/datatreerootitem.h>
#include <picviz/scene.h>
#include <picviz/source.h>
#include <picviz/mapped.h>
#include <picviz/plotted.h>
#include <picviz/view.h>
#include <picviz/layer.h>
#include <picviz/selection.h>
#include <picviz/normalize.h>

//#define LOGTYPE "pcre:bluecoat.level1"
//#define LOGFILE "AAA.log"
/* #define LOGTYPE "pcap" */
/* #define LOGFILE "kraken.pcap" */
#define LOGTYPE "pcre:syslog"
#define LOGFILE "test_petit.log"


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
	printf("Datatree created\n");
	scene = picviz_scene_new(datatree, "default");
	printf("Scene created\n");
	source = picviz_source_new(scene);
	printf("Source created\n");
	picviz_source_file_append(source, LOGTYPE, LOGFILE, NULL);
	printf("Source file append finished\n");
	mapping = picviz_mapping_new(source);
	printf("Mapping created\n");
	mapped = picviz_mapped_make(mapping);
	printf("Mapped created\n");
	plotting = picviz_plotting_new(mapped);
	printf("Plotting created\n");
	plotted = picviz_plotted_build(plotting);
	printf("Plotted created\n");
	retval = picviz_plotted_csv_export(plotted, "lowlevel-plotted.csv");

	view = picviz_view_new(plotted);
	printf("View created\n");
	layer = picviz_layer_new("default");
	//picviz_view_layer_append(view, layer);
	printf("Layer created\n");

	selection = picviz_selection_new();
	picviz_selection_A2A_select_all(selection);
	printf("Selection performed\n");
	picviz_layer_set_selection_by_copy(layer, selection);

	picviz_terminate();

	return 0;
}
