/* Plotting float default */

#include <stdio.h>

#include <picviz/general.h>
#include <picviz/debug.h>

#include <picviz/limits.h>


LibCPPExport float picviz_plotting_function_enumipv4_default(void *plotting, int index, float value, void *userdata, int is_first)
{

	return value;
}


LibCPPExport int picviz_function_init(picviz_function_t *function)
{
	picviz_function_declare(function, PICVIZ_FUNCTION_PLOTTING, "enumipv4", "default");
	picviz_function_set_plotting_function(function, picviz_plotting_function_enumipv4_default);

	return 0;
}


