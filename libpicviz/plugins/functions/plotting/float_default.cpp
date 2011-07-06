/* Plotting float default */

#include <stdio.h>

#include <picviz/general.h>

#include <picviz/limits.h>

#include <picviz/PVPlotting.h>

LibCPPExport float picviz_plotting_exec(const Picviz::PVPlotting_p plotting, PVCol index, float value, void *userdata, bool is_first)
{
	return (value / PICVIZ_MAXFLOAT);
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

