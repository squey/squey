#include <stdio.h>
#include <unistd.h>

#include <picviz/general.h>
#include <picviz/input.h>
#include <picviz/image.h>
#include <picviz/nraw.h>
#include <picviz/plotted.h>
#include <picviz/mapped.h>


#define NOPRINT 1

#define LOGTYPE "pcap"
#define LOGFILE "tcp-extreme.pcap"
/* #define LOGTYPE "syslog" */
/* #define LOGFILE "shortlog" */


void nraw_values_foreach(int index, const char *value, int is_last, void *userdata)
{
#ifndef NOPRINT
	printf("index:%d,value:%s\n", index, value);
#endif
}

int test_table_nraw(picviz_source_t *source)
{
/* 	picviz_table_nraw_t *nraw; */
/* 	picviz_nraw_values_t *nrawvalues; */

/* 	nraw = source->nraw; */

/* 	for (nrawvalues = APR_RING_FIRST(nraw->nraws); nrawvalues != APR_RING_SENTINEL(nraw->nraws, PICVIZ_RING_STRUCTOF(picviz_nraw_values_t), link); nrawvalues = APR_RING_NEXT(nrawvalues, link)) { */
/* #ifndef NOPRINT */
/* 		printf("New line...\n"); */
/* #endif */
/* 		picviz_nraw_values_array_foreach(nrawvalues, nraw_values_foreach, NULL); */
/* 	} */

/* 	picviz_nraw_csv_export(nraw, "local.csv"); */

	return 0;
}

int test_table_selection(void)
{
	picviz_selection_t *tableselection;

	tableselection = picviz_selection_new();
#ifndef NOPRINT
	printf("Line 12345 selection status: '%d'\n", picviz_selection_get_line(tableselection, 12345));
#endif
	picviz_selection_set_line(tableselection, 12345, 1);
#ifndef NOPRINT
	printf("Line 12345 selection status: '%d'\n", picviz_selection_get_line(tableselection, 12345));
#endif

	return 0;
}

int test_layers(picviz_image_t *image)
{
	picviz_layer_t *layer;

	layer = picviz_layer_new(image->pool, "default");
	picviz_image_layer_append(image, layer);
	layer = picviz_layer_new(image->pool, "sshd");
	picviz_image_layer_append(image, layer);
	layer = picviz_layer_new(image->pool, "proftpd");
	picviz_image_layer_append(image, layer);

	return 0;
}

picviz_mapped_t *test_mapping(picviz_source_t *source, char *logtype, char *filename)
{
	picviz_mapped_t *mapping;
	/* picviz_mapping_values_t *mapvals; */

	int i;

	mapping = picviz_mapped_build(source);
	if (!mapping) {
		fprintf(stderr, "Error: Cannot build mapped, exiting!\n");
		exit(1);
	}
/* 	picviz_mapping_debug_columns(mapping); */
/* 	for (mapvals = APR_RING_FIRST(mapping->mappings); mapvals != APR_RING_SENTINEL(mapping->mappings, PICVIZ_RING_STRUCTOF(picviz_mapping_values_t), link); mapvals = APR_RING_NEXT(mapvals, link)) { */
/* #ifndef NOPRINT */
/* 		printf("NEW LINE MAPPING\n"); */
/* #endif */
/* 		for ( i = 0; i < mapvals->values->nelts; i++ ) { */
/* 			float pos = ((float *)mapvals->values->elts)[i]; */
/* #ifndef NOPRINT */
/* 			printf("index:%d;pos:%f\n", i, pos); */
/* #endif */
/* 		} */

/* 	} */

	picviz_mapped_csv_export(mapping, NULL);

	return mapping;
}

int main(int argc, char **argv)
{
	picviz_input_t *input;
	picviz_table_nraw_t *nraw;
        picviz_context_t *context;
	picviz_source_t *source;
        picviz_image_t *image;
	picviz_mapped_t *mapping;
	picviz_plotted_t *plotted;

	context = picviz_init(argc, argv);

	source = picviz_source_file_append(context->pool, LOGTYPE, LOGFILE);

	/* /\* printf("nb rows:%llu\n", source->nblines); *\/ */
	/* picviz_nraw_csv_export(source->nraw, NULL); */
	
	/* test_table_nraw(source); */

/* #ifndef NOPRINT */
/* 	fprintf(stderr, "Guessed logtype: %s\n", source->logtype); */
/* /\* #endif *\/ */

	/* image = picviz_image_new(context, source); */

	/* test_layers(image); */
/* 	test_table_selection(); */

	mapping = test_mapping(source, LOGTYPE, LOGFILE);
	/* test_mapping(image, "iptables", "iptables.log"); */

	plotted = picviz_plotted_build(mapping);
	picviz_plotted_debug(plotted);
	
	/* picviz_debug_image(image); */

	picviz_terminate(context);

	return 0;
}
