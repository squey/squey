#include <picviz/general.h>
#include <picviz/debug.h>

#include <picviz/datatreerootitem.h>
#include <picviz/scene.h>
#include <picviz/source.h>
#include <picviz/nraw.h>

#include <textdig/textdig.h>
#include <textdig/read-file.h>

picviz_source_t *textdig_read_file_get_source(picviz_file_t *file, textdig_options_t options)
{
	picviz_datatreerootitem_t *datatree;
	picviz_scene_t *scene;
	picviz_source_t *source;

	datatree = picviz_datatreerootitem_new();
	scene = picviz_scene_new(datatree, "default");
	source = picviz_source_new(scene);
	if (!source) {
		fprintf(stderr, "*** Error: cannot create source type '%s' file '%s'!\n", options.read_file_type, file->filename);
		return NULL;
	}
	picviz_source_file_append(source, options.read_file_type, file->filename, file);

	return source;
}

void textdig_read_file_source_to_csv(picviz_source_t *source)
{
	picviz_nraw_csv_export((picviz_nraw_t *)source->nraw, NULL);
}

