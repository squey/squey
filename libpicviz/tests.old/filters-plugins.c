#include <stdio.h>

#include <apr_hash.h>

#include <picviz/general.h>
#include <picviz/datatreerootitem.h>
#include <picviz/filters.h>

void filtering_function_foreach(char *name, picviz_filter_t *filter, void *userdata)
{
	printf("Filter plugin name:%s\n", name);
}

int main(int argc, char **argv)
{
	picviz_datatreerootitem_t *datatree;
	apr_hash_index_t *hi;

#include "test-env.h"

	picviz_init(argc, NULL);

	datatree = picviz_datatreerootitem_new();

	picviz_filters_foreach_filter(datatree, filtering_function_foreach, NULL);

	picviz_terminate();

	return 0;
}
