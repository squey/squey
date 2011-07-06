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

/* #define LOGTYPE "pcre:syslog" */
#define LOGFILE "test_syslog.log"
/* #define LOGFILE "kraken.pcap" */

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


	setenv("PICVIZ_PARSERS_DIR","../plugins/parsers/",0);
	setenv("PICVIZ_NORMALIZE_DIR","../plugins/normalize/",0);
	setenv("PICVIZ_FUNCTIONS_DIR","../plugins/functions/",0);

	picviz_init(argc, NULL);

	datatree = picviz_datatreerootitem_new();
	scene = picviz_scene_new(datatree, "default");

	source = picviz_source_new(scene);
	printf("File type:%s\n", picviz_source_filetype_discover(source, LOGFILE));

	picviz_terminate();

	return 0;
}
