#include <picviz/general.h>
#include <picviz/datatreerootitem.h>
#include <picviz/scene.h>
#include <picviz/source.h>
#include <picviz/mapped.h>
#include <picviz/plotted.h>
#include <picviz/view.h>

#define LOGTYPE "pcre:syslog"
#define LOGFILE "test_petit.log"

int main(void)
{
	picviz_datatreerootitem_t *datatree;
	picviz_scene_t *scene;
	picviz_view_t *view;

#include "test-env.h"

	picviz_init(0, NULL);

	datatree = picviz_datatreerootitem_new();
	scene = picviz_scene_new(datatree, "default");

	view = picviz_view_import_file_from_scene(scene, LOGTYPE, LOGFILE);
	/* picviz_view_destroy(view); */
	/* view = picviz_view_import_file_from_scene(scene, LOGTYPE, LOGFILE); */
	/* view = picviz_view_import_file_from_scene(scene, LOGTYPE, LOGFILE); */
	/* view = picviz_view_import_file_from_scene(scene, LOGTYPE, LOGFILE); */

	picviz_view_debug(view);

	picviz_view_destroy(view);

	picviz_terminate();

	return 0;
}
