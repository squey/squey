/* Plotting float default */

#include <stdio.h>

#include <picviz/general.h>
#include <picviz/debug.h>

#include <picviz/limits.h>

#include <picviz/PVPlotting.h>
using Picviz::PVPlotting;

LibCPPExport float picviz_plotting_exec(const PPicviz::VPlotting_p plotting, PVCol index, float value, void *userdata, bool is_first)
{
	return (value / PICVIZ_IPV4_MAXVAL);
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
