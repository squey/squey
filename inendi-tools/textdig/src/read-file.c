/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/general.h>
#include <inendi/debug.h>

#include <inendi/datatreerootitem.h>
#include <inendi/scene.h>
#include <inendi/source.h>
#include <inendi/nraw.h>

#include <textdig/textdig.h>
#include <textdig/read-file.h>

inendi_source_t *textdig_read_file_get_source(inendi_file_t *file, textdig_options_t options)
{
	inendi_datatreerootitem_t *datatree;
	inendi_scene_t *scene;
	inendi_source_t *source;

	datatree = inendi_datatreerootitem_new();
	scene = inendi_scene_new(datatree, "default");
	source = inendi_source_new(scene);
	if (!source) {
		fprintf(stderr, "*** Error: cannot create source type '%s' file '%s'!\n", options.read_file_type, file->filename);
		return NULL;
	}
	inendi_source_file_append(source, options.read_file_type, file->filename, file);

	return source;
}

void textdig_read_file_source_to_csv(inendi_source_t *source)
{
	inendi_nraw_csv_export((inendi_nraw_t *)source->nraw, NULL);
}

