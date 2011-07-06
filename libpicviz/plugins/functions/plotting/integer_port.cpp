/* Plotting float default */

#include <stdio.h>

#include <picviz/general.h>

#include <picviz/limits.h>

#include <picviz/PVPlotting.h>

LibCPPExport float picviz_plotting_exec(const Picviz::PVPlotting_p plotting, pvcol index, float value, void *userdata, bool is_first)
{
	if (value <= 1024) {
		return ((value - 0.5) / 1024);
	} else {
		return ((value / (2*65535)) + 0.5);		
	}
}

LibCPPExport int picviz_plotting_init()
{
	return 0;
}

LibCPPExport int picviz_plotting_terminate()
{
	return 0;
}

LibCPPExport int picviz_plotting_test()
{
	return 0;
}

