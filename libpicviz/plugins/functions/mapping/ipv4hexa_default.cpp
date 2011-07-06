#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <dnet/os.h>
#include <dnet/ip.h>

#include <picviz/general.h>
#include <picviz/debug.h>

#include <picviz/PVMapping.h>

LibCPPExport float picviz_mapping_exec(const Picviz::PVMapping_p mapping, PVCol index, QString &value, void *userdata, bool is_first)
{
	unsigned int intval;
	
	// intval = strtoul(value, NULL, 16);
	intval = value.toUInt();

	return (float)intval;
}


// LibCPPExport int picviz_function_test(void)
// {
// 	char *ipaddr;
// 	float retval;
// 	float expected;

// 	ipaddr = "FF000001";
// 	expected = 2130706432;
// 	retval = picviz_mapping_function_ipv4hexa_default(NULL, 0, ipaddr, NULL, -1);
// 	if (retval != expected) {
// 		picviz_debug(PICVIZ_DEBUG_CRITICAL, "The IP addr '%s' is not mapped correctly expected '%f' but got '%f'!\n", ipaddr, expected, retval);
// 	}

// 	return 0;
// }

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
