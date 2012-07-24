/**
 * \file basic-foreach.c
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <textdig/textdig.h>
#include <textdig/basic-foreach.h>

char *index_filename = NULL;

int print_line_x_function(long offset, char *line, size_t linesize, void *userdata)
{
	textdig_options_t *options = (textdig_options_t *)userdata;

	options->current_line_count++;

	if (options->current_line_count == options->n_line_print) {
	  printf("%s\n", line);
	}


	return 0;
}

int split_lines_function(long offset, char *line, size_t linesize, void *userdata)
{
	textdig_options_t *options = (textdig_options_t *)userdata;

	options->current_line_count++;

	if (offset == 0) {
		/* int index_size; */
		if (index_filename)  free(index_filename);

		options->current_index++;
		index_filename = malloc(strlen(options->current_filename) + 5); /* 5 is enough to have a good index */
		sprintf(index_filename, "%s_0%d", options->current_filename, options->current_index);
		/* printf("indexfilename=%s\n", index_filename); */
		printf("Creating %s...\n", index_filename);
		options->fd = fopen(index_filename, "w");
	}

	fprintf(options->fd, "%s\n", line);

	if (options->current_line_count == options->n_split) {
		options->current_index++;
		fclose(options->fd);
		if (options->current_index < 10) {
			sprintf(index_filename, "%s_0%d", options->current_filename, options->current_index);
		} else {
			sprintf(index_filename, "%s_%d", options->current_filename, options->current_index);
		}
		printf("Creating %s...\n", index_filename);
		options->fd = fopen(index_filename, "w");

		/* printf("%ld;%s\n", offset, line); */
		options->current_line_count = 0;
	}

	return 0;
}
