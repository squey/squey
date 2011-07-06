#include <stdlib.h>

#include <picviz/PVMapping.h>

LibCPPExport float picviz_mapping_exec(const Picviz::PVMapping_p mapping, PVCol index, QString &value, void *userdata, bool is_first)
{
	float fval;

	fval = (float)value.toInt();

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


