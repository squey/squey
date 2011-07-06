/*
 * $Id$
 * Copyright (C) Sebastien Tricaud 2010
 * Copyright (C) Philippe Saade 2010
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <picviz/general.h>
#include <picviz/debug.h>
#include <picviz/view.h>
#include <picviz/layer.h>
#include <picviz/source.h>
#include <picviz/plotted.h>
#include <picviz/selection.h>
#include <picviz/format.h>
#include <picviz/arguments.h>
#include <picviz/filters.h>
#include <picviz/file.h>

LibExport void picviz_filtering_function_init(void)
{
}

LibExport picviz_arguments_t *picviz_filtering_function_get_arguments(void)
{
	return NULL;
}

LibExport picviz_filter_type_t picviz_filtering_function_get_type(void)
{
	return PICVIZ_FILTER_NOCONFIG;
}

LibExport char *picviz_filtering_function_exec(picviz_view_t *view, picviz_layer_t *input_layer, picviz_layer_t *output_layer, picviz_arguments_t *arguments)
{
	char *snort_prefix_cmdline = "C:\\Snort\\bin\\snort.exe -q -A fast -c C:\\Snort\\etc\\snort.conf";
	char *snort_cmdline;

	picviz_source_t *source;
	picviz_source_t *snortsource;
	char *pcap_filename;
	//char *alert_file = ".\\alert.ids";

	char *snort_sip;
	char *pcap_sip;
	char *snort_dip;
	char *pcap_dip;
	char *snort_sport;
	char *pcap_sport;
	char *snort_dport;
	char *pcap_dport;

	int line_id, j;

	picviz_color_t *color;


	source = view->PICVIZ_OBJECT_PLOTTED(parent)->PICVIZ_OBJECT_PLOTTING(parent)->PICVIZ_OBJECT_MAPPED(parent)->PICVIZ_OBJECT_MAPPING(parent)->PICVIZ_OBJECT_SOURCE(parent);

	if (strncmp("pcap", source->logtype, 4)) {
	        picviz_debug(PICVIZ_DEBUG_CRITICAL, "Cannot brush with Snort since the logtype is not a pcap (%s)!\n", source->logtype);
		return "Cannot brush with Snort since the logtype is not a pcap";
	}

	printf("Source TYPE='%s'; FILENAME='%s'\n", source->logtype, source->sourcename);

	pcap_filename = strdup(source->sourcename);

	if (picviz_file_exists("alert.ids")) {
		unlink("alert.ids");
	}

	snort_cmdline = malloc(strlen(snort_prefix_cmdline)+strlen(" -r  -l ./")+strlen(pcap_filename));
	sprintf(snort_cmdline, "%s -r %s -l ./", snort_prefix_cmdline, pcap_filename);
	picviz_debug(PICVIZ_DEBUG_NOTICE, "Snort Plugin Execute :%s\n", snort_cmdline);
	system(snort_cmdline);
//	free(snort_cmdline);
	snortsource = picviz_source_new(source->parent);
	picviz_source_file_append(snortsource, "pcre:snort-fast", "alert.ids", NULL);

	picviz_nraw_flush_row_pos(source->nraw);
	picviz_selection_A2A_select_none(output_layer->selection);
	for (line_id=0; line_id < source->nraw->nraws->nelts; line_id++) {
		pcap_sip = picviz_nraw_get_current_column(source->nraw, 1);
		pcap_dip = picviz_nraw_get_current_column(source->nraw, 2);
		pcap_sport = picviz_nraw_get_current_column(source->nraw, 6);
		pcap_dport = picviz_nraw_get_current_column(source->nraw, 7);

/* 		printf("PCAP: Source IP : '%s', Dest IP : '%s', Sport : '%s', Dport : '%s'\n", pcap_sip, pcap_dip, pcap_sport, pcap_dport); */

		picviz_nraw_flush_row_pos(snortsource->nraw);
		for (j=0; j < snortsource->nraw->nraws->nelts; j++) {
			snort_sip = picviz_nraw_get_current_column(snortsource->nraw, 4);
			snort_dip = picviz_nraw_get_current_column(snortsource->nraw, 6);
			snort_sport = picviz_nraw_get_current_column(snortsource->nraw, 5);
			snort_dport = picviz_nraw_get_current_column(snortsource->nraw, 7);

/* 			printf("SNORT: Source IP : '%s', Dest IP : '%s', Sport : '%s', Dport : '%s'\n", snort_sip, snort_dip, snort_sport, snort_dport); */

			if (!strcmp(pcap_sip, snort_sip)) {
				if (!strcmp(pcap_dip, snort_dip)) {
					if (!strcmp(pcap_sport, snort_sport)) {
						if (!strcmp(pcap_dport, snort_dport)) {
/* 							printf("WE MATCH!\n"); */
							picviz_selection_set_line(output_layer->selection, line_id, 1);
							//color = picviz_color_new();
							//picviz_color_set_rgba(color, 255, 0, 0, 0);
							//picviz_lines_properties_line_set_rgb_from_color(output_layer->lines_properties, line_id, color);
						}
					}
				}
			}
			
			picviz_nraw_next(snortsource->nraw);
		}

		picviz_nraw_next(source->nraw);
	}

	picviz_selection_A2A_select_all(output_layer->selection);

	picviz_debug(PICVIZ_DEBUG_NOTICE, "Execution finished");

	return NULL;
}

LibExport void picviz_filtering_function_terminate(void)
{

}
