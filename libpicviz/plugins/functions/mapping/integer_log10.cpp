#include <stdlib.h>
#include <math.h>

#include <picviz/debug.h>

#include <picviz/PVMapping.h>

int show_debug_limit = 0;

LibCPPExport float picviz_mapping_exec(const Picviz::PVMapping_p mapping, PVCol index, QString &value, void *userdata, bool is_first)
{
	float fval;
	float fval_log10;

	fval = (float)value.toInt();

	if (fval > 0) {
	        return log10f(fval);	  
	}

	if (show_debug_limit < 10) {
		picviz_debug(PICVIZ_DEBUG_CRITICAL, "[limited message] value %s is <= 0. Cannot perform log10, returning 0.\n", value);
		show_debug_limit++;
	}
	return fval;
}

LibCPPExport int picviz_mapping_init()
{
	return 0;
}

LibCPPExport int picviz_mapping_terminate()
{
	return 0;
}

LibCPPExport int picviz_mapping_test()
{
	return 0;
}


