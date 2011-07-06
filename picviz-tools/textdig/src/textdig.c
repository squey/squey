/*
 * $Id$
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>


#include <picviz/general.h>
#include <picviz/debug.h>
#include <picviz/file.h>
#include <picviz/string.h>

#include <picviz/init.h>
#include <picviz/debug.h>
#include <picviz/datatreerootitem.h>
#include <picviz/scene.h>
#include <picviz/source.h>

#include <textdig/textdig.h>
#include <textdig/basic-foreach.h>
#include <textdig/read-file.h>

void print_help(void)
{
	printf("Syntax: textdig [OPTIONS] file1 [file2] [file3] [...] \n\
		  [OPTIONS] can be:\n\
			-c: count the number of lines\n\
			-l: print the file properties (size, ...)\n\
			-px: print the line x\n\
			-sx: split the line every x line\n");
}

void textdig_options_init(textdig_options_t *options)
{
	options->search_char = '\0';
	options->line_count = 0;
	options->current_index = 0;
	options->fd = NULL;
	options->current_filename = NULL;
	options->n_files = 0;
	options->do_count = 0;
	options->n_line_print = 0;
	options->n_split = 0;
	options->files = NULL;
	options->current_line_count = 0;
	options->interactive_console = 0;
	options->list_props = 0;
	options->read_file = 0;
	options->read_file_type = "automatic";
	options->read_file_name = NULL;
	options->output_csv = 0;
}

void textdig_options_print(textdig_options_t options)
{
	int i;
  
	printf("do_count=%d\n", options.do_count);
	printf("n_line_print=%ld\n", options.n_line_print);
	printf("n_split=%ld\n", options.n_split);
	printf("n_files=%d\n", options.n_files);
	printf("Files: ");
	for (i = 0; i < options.n_files; i++) {
		printf("\t%d - %s\n", i, options.files[i]);
	}
}

int main(int argc, char **argv)
{
	int c;
	textdig_options_t options;
	picviz_file_t *file;
	picviz_source_t *source;
	int i;
	int ret;


	picviz_init(0, NULL);

	if (argc < 2) {
		print_help();
		return 1;
	}

	textdig_options_init(&options);

	while ((c = getopt(argc, argv, "cilp:e:ms:t:o")) != -1)
	  switch (c)
	    {
	    case 'c':
	      options.do_count = 1;
	      break;
	    case 'i':
	      options.interactive_console = 1;
	      break;
	    case 'l':
	      options.list_props = 1;
	      break;
	    case 'o':
	      options.output_csv = 1;
	      break;
	    case 'p':
	      options.n_line_print = atoi(optarg);
	      break;
	    case 't':
              options.read_file_type = optarg;
	      break;
	    /* case 'r': */
	    /*   options.read_file = 1; */
            /*   options.read_file_name = optarg; */
	    /*   break; */
	    case 's':
	      options.n_split = atoi(optarg);
	      break;
	    default:
	      print_help();
	      return 1;
	    }

	if (!options.interactive_console) {

		options.n_files = argc - optind;
		options.files = argv + optind;

		if (options.n_files) {
		/* textdig_options_print(options); */
		for (i=0; i < options.n_files; i++) {
			picviz_debug(PICVIZ_DEBUG_NOTICE, "Processing %s...\n", options.files[i]);

			file = picviz_file_new(options.files[i]);
			options.current_line_count = 0;
			options.current_index = 0;
			options.current_filename = options.files[i];

			if (options.do_count) {
				printf("lines:%ld\n", file->nblines);
			}

			if (options.n_line_print) {
		        	ret = picviz_file_line_foreach(file, print_line_x_function, &options);
			}

			if (options.n_split) {
		        	ret = picviz_file_line_foreach(file, split_lines_function, &options);
	        		fclose(options.fd);
			}

			if (options.list_props) {
			  printf("size:%ld\n", file->size);
			}

			if (options.output_csv) {
			  source = textdig_read_file_get_source(file, options);
			  textdig_read_file_source_to_csv(source);
			}

			picviz_file_destroy(file);

			picviz_debug(PICVIZ_DEBUG_NOTICE, "Done!\n");

		}
		} else {	/* if(options.n_files) */
		  /* Nothing to do now but who knows */
			/* if (options.read_file) { */
			/*   picviz_source_t *source; */
			/*   source =  textdig_read_file_get_source(file, options); */
			/*   textdig_read_file_source_to_csv(source); */
			/* } */
       		}
	} else { 		/* options.interactive_console */
		ret = interactive_console_start(options, argc, argv);
		if (!ret) {
			picviz_debug(PICVIZ_DEBUG_CRITICAL, "Cannot start the interractive console!\n");
			picviz_terminate();
			return 1;
		}
	}

	picviz_terminate();

	return 0;
}
