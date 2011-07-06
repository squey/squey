#include <stdio.h>

#include <picviz/init.h>
#include <picviz/area.h>
#include <picviz/context.h>
#include <picviz/view.h>
#include <picviz/utils.h>

#define LOGTYPE "syslog"
#define LOGFILE "messages"

int main(int argc, char **argv)
{
	picviz_context_t *context;
	picviz_view_t *view;
	picviz_selection_t *selection;

	context = picviz_init(argc, argv);
	view = picviz_h_view_create(context, LOGTYPE, LOGFILE);
	picviz_selection_select_all(view->selection);

	printf("Only libvirtd should appear since we select from 0.1 to 0.4 (y) and between axis 1 and 2 (x):\n");
	view->selection = picviz_area_get_square_selection(view, 1.1, 1.9, 0.1, 0.4);
	picviz_view_foreach_selected_lines_nraw(view, picviz_array_default_print_callback, NULL);

	picviz_selection_select_all(view->selection);

	printf("Everything BUT libvirtd should appear since we select from 0.7 to 0.9 (y) and between axis 2 and 3 (x):\n");
	view->selection = picviz_area_get_square_selection(view, 1.1, 2.8, 0.6, 0.9);
	picviz_view_foreach_selected_lines_nraw(view, picviz_array_default_print_callback, NULL);

	return 0;
}
