#include <picviz/general.h>
#include <picviz/debug.h>
#include <picviz/file.h>
#include <picviz/string.h>

#include <picviz/init.h>
#include <picviz/datatreerootitem.h>
#include <picviz/scene.h>
#include <picviz/source.h>

int main(void)
{
	picviz_datatreerootitem_t *datatree;
	picviz_scene_t *scene;
	picviz_source_t *source;
	
	picviz_file_t *file;


#include "test-env.h"

	picviz_init(0, NULL);

	file = picviz_file_new("test_petit.log");

        datatree = picviz_datatreerootitem_new();
        scene = picviz_scene_new(datatree, "default");
        source = picviz_source_new(scene);
        picviz_source_file_append(source, "automatic", NULL, file);

	picviz_file_destroy(file);

	picviz_terminate();

	return 0;
}
