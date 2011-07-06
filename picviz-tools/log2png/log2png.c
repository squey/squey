#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>

#include <cairo.h>

#include <picviz/general.h>
#include <picviz/init.h>
#include <picviz/debug.h>
#include <picviz/file.h>
#include <picviz/string.h>
#include <picviz/datatreerootitem.h>
#include <picviz/scene.h>
#include <picviz/source.h>
#include <picviz/view.h>

#include <picviz/mapping.h>
#include <picviz/mapped.h>
#include <picviz/plotting.h>
#include <picviz/plotted.h>

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

picviz_source_t *source_create(void)
{
	picviz_datatreerootitem_t *datatree;
	picviz_scene_t *scene;
	picviz_source_t *source;

	datatree = picviz_datatreerootitem_new();
	scene = picviz_scene_new(datatree, "default");
	source = picviz_source_new(scene);

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


void write_png(picviz_file_t *file, char *output_png)
{
	picviz_source_t *source;
	picviz_mapping_t *mapping;
	picviz_mapped_t *mapped;
	picviz_plotting_t *plotting;
	picviz_plotted_t *plotted;
	picviz_view_t *view;

	cairo_surface_t *surface;
	cairo_t *cr;

	int i;
	int colpos;
	double x,y;

	picviz_line_properties_t line_properties;

	picviz_debug(PICVIZ_DEBUG_DEBUG, "Write png to '%s'\n", output_png);

	source = source_create();
	picviz_source_file_append(source, "automatic", NULL, file);

	mapping = picviz_mapping_new(source);
	mapped = picviz_mapped_make(mapping);
	plotting = picviz_plotting_new(mapped);
	plotted = picviz_plotted_build(plotting);

	view = picviz_view_new(plotted);

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
		   draw_text(cr, picviz_view_axis_get_name(view, i), 20, (double)(i * 400) + PADDING, 25 );
		} else {
			if (i== source->column_count) {
				draw_text(cr, picviz_view_axis_get_name(view, i), 20, (double)(i * 400) + PADDING - 15, 25 );
			} else {
				draw_text(cr, picviz_view_axis_get_name(view, i), 20, (double)(i * 400) + PADDING - 30, 25 );
			}
		}
	}
	cairo_stroke(cr);

	picviz_view_process_from_layer_stack(view);
        picviz_view_apply_filter_from_name(view, "log_heatline_color", NULL);
        picviz_view_process_from_eventline(view);

	/* Draw the lines */
        cairo_set_line_width (cr, 0.5);
	for (i=0; i < picviz_nraw_count_row(source->nraw); i++) {
		colpos = 0;

		line_properties = picviz_lines_properties_get_line_properties(view->output_layer->lines_properties, i);
		cairo_set_source_rgb(cr, line_properties.r, line_properties.g, line_properties.b);

		while (colpos < (source->column_count - 1)) {
			x = (colpos * 400) + PADDING;
			y = (1 - picviz_plotted_get_value(plotted, i, colpos)) * get_height();
			cairo_move_to(cr, x, y);
			x = ((colpos + 1) * 400) + PADDING;
			y = (1 - picviz_plotted_get_value(plotted, i, colpos + 1)) * get_height();
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
	picviz_file_t *file = NULL;
	char *png_output = NULL;
	char c;

	picviz_init(0, NULL);

	if ((argc < 2)||(argv[1][0] == '-')) {
		print_help();
		return 1;
	}

	picviz_debug(PICVIZ_DEBUG_NOTICE, "Starting log2png\n");

	file = picviz_file_new(argv[1]);
	picviz_debug(PICVIZ_DEBUG_NOTICE, "File Acquired\n");

	if (argc == 2) {
		picviz_file_destroy(file);
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

	picviz_debug(PICVIZ_DEBUG_NOTICE, "Finished\n");
	picviz_file_destroy(file);
	picviz_terminate();

	return 0;
}
