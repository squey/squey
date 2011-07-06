#include <stdio.h>

#if TIME_WITH_SYS_TIME
# include <sys/time.h>
# include <time.h>
#else
# if HAVE_SYS_TIME_H
#  include <sys/time.h>
# else
#  include <time.h>
# endif
#endif

#include <pvrush/PVFormat.h>

#include <picviz/general.h>

#include <picviz/limits.h>

#include <picviz/PVMapped.h>
#include <picviz/PVMapping.h>
#include <picviz/PVPlotting.h>

using Picviz::PVPlotting;

LibCPPExport float picviz_plotting_exec(const Picviz::PVPlotting_p plotting, PVCol index, float value, void *userdata, bool is_first)
{
	QString modemapping;
	float retval;
	float ymin, ymax;
	// picviz_plotting_t *plotting_o = (picviz_plotting_t *)plotting;

	PVRush::PVFormat *format = plotting->get_format();

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
	
	// modemapping = picviz_format_column_get_modemapping(format, index);
	modemapping = format->axes[index]["mapping"];

	if (!modemapping.compare("default")) {
		retval = (float) (retval / 9999999999); // TIME_T_MAX;
		return retval;
	} else
	if (!modemapping.compare("24h")) {
		retval = (float) (retval / PICVIZ_TIME_24H_MAX);
		PVLOG_HEAVYDEBUG("Plotting time 24h, retval = '%f'\n", retval);
		return retval;
	}
	else
	if (!modemapping.compare("week")) {
		retval = (float) (retval / PICVIZ_TIME_WEEK_MAX);
		return retval;
	}
	else
	if (!modemapping.compare("month")) {
		retval = (float) (retval / PICVIZ_TIME_MONTH_MAX);
		return retval;
	}
//	else
//	if (!modemapping.compare("epoch")) {
//		// Epoch can be negative !!
//		// FIXME:
//		// It has been made from (potentially) 64-bit signed integers, but we can't divide by LLONG_MAX because it will
//		// always return 0... So, we divide using INT_MAX, but after 2038, the float will be > 1 !!!!
//		retval = (float) ( ((double) retval + (double) INT_MAX ) / ((double)2 * (double)INT_MAX + (double)1) );
//		PVLOG_INFO("%s: time_epoch_plotting: org retval = %f, return retval = %f\n", __FUNCTION__, retval, retval);
//		return retval;
//	}

	fprintf(stderr, "*** Warning(%s): no such mapping mode '%s'. Plotting cannot guess the limit! Dividing by 1.\n", __FILE__, qPrintable(modemapping));
	return (float) (retval / 1);
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
