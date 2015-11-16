/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <stdio.h>

#include <inendi/general.h>
#include <inendi/init.h>
#include <inendi/context.h>
#include <inendi/nraw.h>
#include <inendi/source.h>
#include <inendi/mapped.h>

int main(int argc, char **argv)
{
	inendi_context_t *context;
	inendi_source_t *source;
	inendi_mapped_t *mapped;

	char *logtype;
	char *logfile;

	if (argc < 3) {
		fprintf(stderr, "Syntax error:\n");
		fprintf(stderr, "%s logtype logfile\n", argv[0]);
		return 1;
	}

	logtype = argv[1];
	logfile = argv[2];

	context = inendi_init(argc, argv);
	if (!context) {
		fprintf(stderr, "*** Error: cannot initialize library context!\n");
		return 1;
	}

#if 0
	source = inendi_source_file_append(source, logtype, logfile);
	if (!source) {
		fprintf(stderr, "*** Error: cannot create source type '%s' file '%s'!\n", logtype, logfile);
		return 1;
	}

	mapped = inendi_mapped_build(source);
	if (!mapped) {
		fprintf(stderr, "Error: Cannot build mapped, exiting!\n");
		exit(1);
	}

	inendi_mapped_csv_export(mapped, NULL);
#endif

	inendi_terminate(context);

	return 0;
}
