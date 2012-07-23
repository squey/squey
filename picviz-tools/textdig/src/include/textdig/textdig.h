/**
 * \file textdig.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef _TEXTDIG_H_
#define _TEXTDIG_H_

#include <stdio.h>

#ifdef __cplusplus
 extern "C" {
#endif

#define TEXTDIG_VER_MAJOR 0
#define TEXTDIG_VER_MINOR 1

struct _textdig_options_t {
	int do_count;
  	long n_line_print;
	long current_line_count;
	long n_split;
	char search_char;
	long line_count;
	int split_x;
	int current_index;
	FILE *fd;
	char **files;
	char *current_filename;
	/* char *index_filename; */
	int n_files;		/* How many files should we treat? */
	int interactive_console;
        int list_props;
        int read_file;
        char *read_file_type;
        char *read_file_name;
        int output_csv;
};
typedef struct _textdig_options_t textdig_options_t;


#ifdef __cplusplus
 }
#endif


#endif /* _TEXTDIG_H_ */

