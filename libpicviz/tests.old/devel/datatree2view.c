#include <stdio.h>

#include <picviz/init.h>
#include <picviz/datatreerootitem.h>
#include <picviz/view.h>

#define LOGTYPE "pcap"
#define LOGFILE "kraken.pcap"

int main(int argc, char **argv)
{
	picviz_context_t *context;
	picviz_datatreerootitem_t *datatree;
	picviz_scene_t *scene;
	picviz_source_t *source;

	context = picviz_init(argc, argv);

	datatree = picviz_datatreerootitem_new();
	scene = picviz_scene_new(datatree, "default");
	picviz_datatreerootitem_scene_append(datatree, scene);

	source = picviz_source_file_append(context->pool, LOGTYPE, LOGFILE);
	picviz_scene_source_append(scene, source);

	picviz_terminate(context);

	return 0;
}
