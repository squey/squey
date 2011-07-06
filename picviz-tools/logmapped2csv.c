#include <stdio.h>

#include <picviz/general.h>
#include <picviz/init.h>
#include <picviz/context.h>
#include <picviz/nraw.h>
#include <picviz/source.h>
#include <picviz/mapped.h>

int main(int argc, char **argv)
{
	picviz_context_t *context;
	picviz_source_t *source;
	picviz_mapped_t *mapped;

	char *logtype;
	char *logfile;

	if (argc < 3) {
		fprintf(stderr, "Syntax error:\n");
		fprintf(stderr, "%s logtype logfile\n", argv[0]);
		return 1;
	}

	logtype = argv[1];
	logfile = argv[2];

	context = picviz_init(argc, argv);
	if (!context) {
		fprintf(stderr, "*** Error: cannot initialize library context!\n");
		return 1;
	}

#if 0
	source = picviz_source_file_append(source, logtype, logfile);
	if (!source) {
		fprintf(stderr, "*** Error: cannot create source type '%s' file '%s'!\n", logtype, logfile);
		return 1;
	}

	mapped = picviz_mapped_build(source);
	if (!mapped) {
		fprintf(stderr, "Error: Cannot build mapped, exiting!\n");
		exit(1);
	}

	picviz_mapped_csv_export(mapped, NULL);
#endif

	picviz_terminate(context);

	return 0;
}
