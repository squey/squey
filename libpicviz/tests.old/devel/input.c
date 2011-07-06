#include <picviz/context.h>
#include <picviz/general.h>
#include <picviz/input.h>
#include <picviz/nraw.h>

int main(int argc, char **argv)
{
        picviz_context_t *context;
	picviz_input_t *input;
	picviz_table_nraw_t *nraw;
	
	context = picviz_init(argc, argv);

	input = picviz_input_new_from_logtype("syslog");
	/* picviz_input_debug(input); */

	nraw = picviz_normalize_new(context->pool, input, "shortlog");

	picviz_terminate(context);

	return 0;
}
