/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>

#include <cairo.h>

#include <inendi/general.h>
#include <inendi/init.h>
#include <inendi/debug.h>
#include <inendi/file.h>
#include <inendi/string.h>
#include <inendi/datatreerootitem.h>
#include <inendi/scene.h>
#include <inendi/source.h>
#include <inendi/view.h>

#include <inendi/mapping.h>
#include <inendi/mapped.h>
#include <inendi/plotting.h>
#include <inendi/plotted.h>

#define PADDING 20

int get_width(int col)
{
	int i;

	i = 400 * (col - 1) + (2 * PADDING);

	return i;
}

int get_height(void)
{
	return 1000;
}

void print_help(void)
{
	printf("Syntax: textdig filename [OPTIONS]\n\
		  [OPTIONS] can be:\n\
			-o file: output png to 'file'\n");
}

inendi_source_t *source_create(void)
{
	inendi_datatreerootitem_t *datatree;
	inendi_scene_t *scene;
	inendi_source_t *source;

	datatree = inendi_datatreerootitem_new();
	scene = inendi_scene_new(datatree, "default");
	source = inendi_source_new(scene);

	return source;
}

void draw_text(cairo_t *cr, char *text, double size, double x, double y)
{
        cairo_set_source_rgb(cr, 1, 1, 1);

        cairo_select_font_face (cr, "Sans", CAIRO_FONT_SLANT_NORMAL,
				CAIRO_FONT_WEIGHT_BOLD);
        cairo_set_font_size (cr, size);

	cairo_move_to (cr, x, y);
	cairo_show_text (cr, text);

        cairo_stroke(cr);
}


void write_png(inendi_file_t *file, char *output_png)
{
	inendi_source_t *source;
	inendi_mapping_t *mapping;
	inendi_mapped_t *mapped;
	inendi_plotting_t *plotting;
	inendi_plotted_t *plotted;
	inendi_view_t *view;

	cairo_surface_t *surface;
	cairo_t *cr;

	int i;
	int colpos;
	double x,y;

	inendi_line_properties_t line_properties;

	inendi_debug(INENDI_DEBUG_DEBUG, "Write png to '%s'\n", output_png);

	source = source_create();
	inendi_source_file_append(source, "automatic", NULL, file);

	mapping = inendi_mapping_new(source);
	mapped = inendi_mapped_make(mapping);
	plotting = inendi_plotting_new(mapped);
	plotted = inendi_plotted_build(plotting);

	view = inendi_view_new(plotted);

	surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, get_width(source->column_count), get_height());

	cr = cairo_create(surface);
	cairo_rectangle (cr, 0, 0, (int)get_width(source->column_count), (int)get_height());
        cairo_set_source_rgba (cr, 0.2, 0.2, 0.2, 1);
        cairo_fill (cr);

        cairo_set_line_width (cr, 1.0);

	/* Draw the axes */
	for (i=0; i < source->column_count; i++) {
		cairo_set_source_rgb(cr, 1, 1, 1);
		cairo_move_to(cr, (double)(i * 400) + PADDING, (double)0);
		cairo_line_to(cr, (double)(i * 400) + PADDING, (double)get_height());
		
		if (i==0) {
		   draw_text(cr, inendi_view_axis_get_name(view, i), 20, (double)(i * 400) + PADDING, 25 );
		} else {
			if (i== source->column_count) {
				draw_text(cr, inendi_view_axis_get_name(view, i), 20, (double)(i * 400) + PADDING - 15, 25 );
			} else {
				draw_text(cr, inendi_view_axis_get_name(view, i), 20, (double)(i * 400) + PADDING - 30, 25 );
			}
		}
	}
	cairo_stroke(cr);

	inendi_view_process_from_layer_stack(view);
        inendi_view_apply_filter_from_name(view, "log_heatline_color", NULL);
        inendi_view_process_from_eventline(view);

	/* Draw the lines */
        cairo_set_line_width (cr, 0.5);
	for (i=0; i < inendi_nraw_count_row(source->nraw); i++) {
		colpos = 0;

		line_properties = inendi_lines_properties_get_line_properties(view->output_layer->lines_properties, i);
		cairo_set_source_rgb(cr, line_properties.r, line_properties.g, line_properties.b);

		while (colpos < (source->column_count - 1)) {
			x = (colpos * 400) + PADDING;
			y = (1 - inendi_plotted_get_value(plotted, i, colpos)) * get_height();
			cairo_move_to(cr, x, y);
			x = ((colpos + 1) * 400) + PADDING;
			y = (1 - inendi_plotted_get_value(plotted, i, colpos + 1)) * get_height();
			cairo_line_to(cr, x, y);

			colpos++;
		}
		cairo_stroke(cr);
	}



	cairo_surface_write_to_png(surface, output_png);

	printf("Wrote to %s\n", output_png);
	cairo_destroy (cr);
        cairo_surface_destroy (surface);

}

int main(int argc, char **argv)
{
	inendi_file_t *file = NULL;
	char *png_output = NULL;
	char c;

	inendi_init(0, NULL);

	if ((argc < 2)||(argv[1][0] == '-')) {
		print_help();
		return 1;
	}

	inendi_debug(INENDI_DEBUG_NOTICE, "Starting log2png\n");

	file = inendi_file_new(argv[1]);
	inendi_debug(INENDI_DEBUG_NOTICE, "File Acquired\n");

	if (argc == 2) {
		inendi_file_destroy(file);
		return 0;
	}

	while ((c = getopt(argc, argv, "o:")) != -1)
	  switch (c)
	    {
	    case 'o':
	      png_output = optarg;
	      break;
	    default:
	      print_help();
	      return 1;
	    }

	write_png(file, png_output);

	inendi_debug(INENDI_DEBUG_NOTICE, "Finished\n");
	inendi_file_destroy(file);
	inendi_terminate();

	return 0;
}
