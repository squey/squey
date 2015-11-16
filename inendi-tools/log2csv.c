/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <stdio.h>

#include <inendi/general.h>
#include <inendi/init.h>
#include <inendi/datatreerootitem.h>
#include <inendi/scene.h>
#include <inendi/source.h>
#include <inendi/nraw.h>

int main(int argc, char **argv)
{
	inendi_source_t *source;
	inendi_datatreerootitem_t *datatree;
	inendi_scene_t *scene;

	char *logtype;
	char *logfile;

	/* setenv("INENDI_DEBUG_LEVEL","QUIET", 0); */

	if (argc < 2) {
		fprintf(stderr, "Syntax error:\n");
		fprintf(stderr, "%s [logtype] logfile\n", argv[0]);
		return 1;
	}

	if (argc == 2) {
		logtype = "automatic";
		logfile = argv[1];
	} else {
		logtype = argv[1];
		logfile = argv[2];
	}

	/* printf("Starting with logtype '%s' and logfile '%s'\n", logtype, logfile); */

	inendi_init(argc, argv);
	datatree = inendi_datatreerootitem_new();
	scene = inendi_scene_new(datatree, "default");
	source = inendi_source_new(scene);
	if (!source) {
		fprintf(stderr, "*** Error: cannot create source type '%s' file '%s'!\n", logtype, logfile);
		return 1;
	}
	inendi_source_file_append(source, logtype, logfile);

	inendi_nraw_csv_export((inendi_nraw_t *)source->nraw, NULL);


	return 0;
}
