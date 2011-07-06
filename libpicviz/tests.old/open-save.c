#include <picviz/general.h>
#include <picviz/datatreerootitem.h>
#include <picviz/scene.h>
#include <picviz/view.h>
#include <picviz/layer.h>
#include <picviz/open-save.h>

#define LOGTYPE "pcre:syslog"
#define LOGFILE "test_petit.log"

int main(void)
{
	picviz_datatreerootitem_t *datatree;
	picviz_scene_t *scene;
	picviz_view_t *view;
	picviz_layer_t *layer;
	picviz_view_t *read_view;

	int retval;
	char *filename;
	
#include "test-env.h"

	filename = "test.pcv";

	picviz_init(0, NULL);

	retval = picviz_open_is_picviz_type(filename);
	printf("Picviz type = %d\n", retval);

	datatree = picviz_datatreerootitem_new();
	scene = picviz_scene_new(datatree, "default");
	view = picviz_view_import_file_from_scene(scene, LOGTYPE, LOGFILE);
	layer = picviz_layer_new("Layer 2");	
	picviz_layer_stack_append_layer(view->layer_stack, layer);

	picviz_view_debug(view);

	picviz_save(view, filename);

	picviz_view_destroy(view);

	read_view = picviz_open(filename);

	picviz_layer_stack_debug(read_view->layer_stack);

	picviz_terminate();

	return 0;

}
