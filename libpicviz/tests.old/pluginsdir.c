#include <stdio.h>

#include <picviz/init.h>
#include <picviz/plugins.h>

int main(int argc, char **argv)
{
	char *list = NULL;

#include "test-env.h"

	printf("PICVIZ_NORMALIZE_DIR=%s\n", picviz_plugins_get_normalize_dir());
	printf("PICVIZ_NORMALIZE_HELPERS_DIR=%s\n", picviz_plugins_get_normalize_helpers_dir());
	printf("PICVIZ_FUNCTIONS_DIR=%s\n", picviz_plugins_get_functions_dir());
	printf("PICVIZ_FILTERS_DIR=%s\n", picviz_plugins_get_filters_dir());

	list = picviz_plugins_normalize_all_list();
	printf("list='%s'\n", list);
	free(list);

	return 0;
}
