#include <picviz/general.h>
#include <picviz/debug.h>
#include <picviz/datatreerootitem.h>

int main(int argc, char **argv)
{
	picviz_datatreerootitem_t *datatree;
	int i;

#include "test-env.h"

	picviz_init(argc, NULL);

	datatree = picviz_datatreerootitem_new();

	/* for ( i = 0; i < datatree->functions->nelts; i++ ) { */
	/* 	picviz_function_t *function = ((picviz_function_t **)datatree->functions->elts)[i]; */
	/* 	if (!function->test_func) { */
	/* 		picviz_debug(PICVIZ_DEBUG_WARNING, "No test function for '%s' mode '%s'!\n", function->name, function->mode); */
	/* 	} else { */
	/* 		picviz_debug(PICVIZ_DEBUG_NOTICE, "TESTING '%s' mode '%s'... ", function->name, function->mode); */
	/* 		if (function->test_func()) { */
	/* 			printf("FAILED!\n"); */
	/* 			return 1; */
	/* 		} else { */
	/* 			printf("PASSED!\n"); */
	/* 		} */
	/* 	} */
	/* } */

	picviz_terminate();

	return 0;
}

