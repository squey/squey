#include <stdio.h>

#include <apr_tables.h>

#include <picviz/general.h>
#include <picviz/datatreerootitem.h>
#include <picviz/scene.h>
#include <picviz/source.h>
#include <picviz/mapping.h>
#include <picviz/mapped.h>
#include <picviz/plotting.h>
#include <picviz/plotted.h>

#define LOGTYPE "csv"
#define LOGFILE "test-all.csv"

int main(int argc, char **argv)
{
	picviz_datatreerootitem_t *datatree;
	picviz_scene_t *scene;
	picviz_source_t *source;
	picviz_mapping_t *mapping;
	picviz_mapped_t *mapped;
	picviz_plotting_t *plotting;
	picviz_plotted_t *plotted;

	setenv("PICVIZ_NORMALIZE_DIR","../plugins/normalize/",0);
	setenv("PICVIZ_FUNCTIONS_DIR","../plugins/functions/",0);

	picviz_init(argc, argv);

	datatree = picviz_datatreerootitem_new();
	scene = picviz_scene_new(datatree, "default");

	source = picviz_source_new(scene);
	picviz_source_file_append(source, LOGTYPE, LOGFILE);
	mapping = picviz_mapping_new(source);
	mapped = picviz_mapped_make(mapping);
	plotting = picviz_plotting_new(mapped);
	plotted = picviz_plotted_build(plotting);

	if (strcmp("time", picviz_format_column_get_type(source->format, 0))) {
		fprintf(stderr, "Error, the type for this column must be time!\n");
		return 1;
	}

	if (strcmp("ipv4", picviz_format_column_get_type(source->format, 1))) {
		fprintf(stderr, "Error, the type for this column must be ipv4!\n");
		return 1;
	}

	if (strcmp("integer", picviz_format_column_get_type(source->format, 2))) {
		fprintf(stderr, "Error, the type for this column must be integer!\n");
		return 1;
	}

	if (strcmp("float", picviz_format_column_get_type(source->format, 3))) {
		fprintf(stderr, "Error, the type for this column must be float!\n");
		return 1;
	}

	if (strcmp("string", picviz_format_column_get_type(source->format, 4))) {
		fprintf(stderr, "Error, the type for this column must be string!\n");
		return 1;
	}

	picviz_terminate();

	return 0;
}
