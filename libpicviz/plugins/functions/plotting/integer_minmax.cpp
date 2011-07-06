/* Plotting float default */

#include <stdio.h>

#include <picviz/general.h>

#include <picviz/limits.h>

#include <picviz/PVMapped.h>
#include <picviz/PVPlotting.h>
#include <picviz/PVMapping.h>

LibCPPExport float picviz_plotting_exec(const Picviz::PVPlotting_p plotting, PVCol index, float value, void *userdata, bool is_first)
{
	float ymin, ymax;
	float retval;
	
	Picviz::mandatory_param_map const& mand_params = plotting->mapped->mapping->get_mandatory_params_for_col(index);
	Picviz::mandatory_param_map::const_iterator it_min = mand_params.find(Picviz::mandatory_ymin);
	Picviz::mandatory_param_map::const_iterator it_max = mand_params.find(Picviz::mandatory_ymax);
	if (it_min == mand_params.end() || it_max == mand_params.end()) {
		PVLOG_WARN("ymin and/or ymax don't exist for axis %d. Maybe the mandatory minmax mapping hasn't be run ?\n", index);
		return value;
	}
	ymin = (*it_min).second.second;
	ymax = (*it_max).second.second;
	
	if (ymin == ymax) return 0.5;

	retval = (value - ymin) / (ymax - ymin);

	return retval;
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
