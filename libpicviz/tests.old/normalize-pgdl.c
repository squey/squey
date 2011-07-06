#include <stdio.h>
#include <picviz/context.h>
#include <picviz/init.h>
#include <picviz/normalize.h>

int main(int argc, char **argv)
{

        picviz_init(argc, NULL);

	setenv("PICVIZ_NORMALIZE_DIR","../plugins/normalize/",0);
	/* picviz_normalize_plugin_register_all(context->pool, NULL); */
	/* picviz_normalize_plugin_load(context->pool, "pgdl"); */

	/* picviz_terminate(context); */

	return 0;
}
