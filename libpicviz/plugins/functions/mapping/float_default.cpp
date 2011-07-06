#include <stdlib.h>

#include <pvcore/debug.h>

#include <picviz/general.h>

#include <picviz/PVMapping.h>

LibCPPExport float picviz_mapping_exec(const Picviz::PVMapping_p mapping, PVCol index, QString &value, void *userdata, bool is_first)
{
	return value.toFloat();
  // float retval;

  // retval = atof(value);

  // PVCore::log(PVCore::loglevel::debug, "%s: retval=%f\n", __FUNCTION__, retval);

  // return retval;
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

