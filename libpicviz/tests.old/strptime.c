#include <stdio.h>
#include <unistd.h>

#include <picviz/general.h>
#include <picviz/init.h>
#include <picviz/datatreerootitem.h>
#include <picviz/scene.h>
#include <picviz/source.h>
#include <picviz/mapping.h>
#include <picviz/mapped.h>
#include <picviz/plotted.h>
#include <picviz/view.h>

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

#include "test-env.h"

	picviz_init(argc, NULL);

	datatree = picviz_datatreerootitem_new();
	scene = picviz_scene_new(datatree, "default");
	source = picviz_source_new(scene);
	/* picviz_source_file_append(source, "csv", "strptime.csv"); */
	picviz_source_file_append(source, "pcre:syslog", "test_syslog.log", NULL);
	mapping = picviz_mapping_new(source);
	mapped = picviz_mapped_make(mapping);
	plotting = picviz_plotting_new(mapped);
	plotted = picviz_plotted_build(plotting);
	view = picviz_view_new(plotted);

	picviz_terminate();

	return 0;
}
