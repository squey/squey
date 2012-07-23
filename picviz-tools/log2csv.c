/**
 * \file log2csv.c
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <stdio.h>

#include <picviz/general.h>
#include <picviz/init.h>
#include <picviz/datatreerootitem.h>
#include <picviz/scene.h>
#include <picviz/source.h>
#include <picviz/nraw.h>

int main(int argc, char **argv)
{
	picviz_source_t *source;
	picviz_datatreerootitem_t *datatree;
	picviz_scene_t *scene;

	char *logtype;
	char *logfile;

	/* setenv("PICVIZ_DEBUG_LEVEL","QUIET", 0); */

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

	picviz_init(argc, argv);
	datatree = picviz_datatreerootitem_new();
	scene = picviz_scene_new(datatree, "default");
	source = picviz_source_new(scene);
	if (!source) {
		fprintf(stderr, "*** Error: cannot create source type '%s' file '%s'!\n", logtype, logfile);
		return 1;
	}
	picviz_source_file_append(source, logtype, logfile);

	picviz_nraw_csv_export((picviz_nraw_t *)source->nraw, NULL);


	return 0;
}
