#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <dnet/os.h>
#include <dnet/ip.h>

#include <picviz/general.h>

#include <picviz/PVMapping.h>

LibCPPExport float picviz_mapping_exec(const Picviz::PVMapping_p mapping, PVCol index, QString &value, void *userdata, bool is_first)
{
	char *buffer_org = NULL, *buffer;
	uint32_t intval = 0;
	int count = 2;
	
	if (value.isEmpty()) {
		PVLOG_ERROR("%s: Cannot get the string value! Returns 0\n", __FUNCTION__);
		return 0;
	}

	buffer_org = strdup(value.toUtf8().data());
	buffer = buffer_org;

	buffer = strchr(buffer, '.');
	if (!buffer) {
		PVLOG_ERROR("%s: ipv4_mapping: IPv4 address %s has an invalid format. Returns 0\n", __FUNCTION__, qPrintable(value));
		return 0;
	}
	*buffer = 0;
	intval = ((uint32_t) atoi(buffer_org)) << 24;
	buffer++;
	char* buffer_prev = buffer;
	while(count > 0) {
		buffer = strchr(buffer_prev, '.');
		if (!buffer) {
			PVLOG_ERROR("%s: ipv4_mapping: IPv4 address %s has an invalid format. Returns 0\n", __FUNCTION__, qPrintable(value));
			return 0;
		}
		*buffer = 0;
		intval |= ((uint32_t)atoi(buffer_prev)) << count*8;
		buffer_prev = buffer+1;
		count--;
	}
	intval |= (uint32_t) atoi(buffer_prev);

	free(buffer_org);

	return (float)intval;
}



// LibCPPExport int picviz_function_test(void)
// {
// 	char *ipaddr;
// 	float retval;
// 	float expected;

// 	ipaddr = "127.0.0.1";
// 	expected = 2130706432;
// 	retval = picviz_mapping_function_ipv4_default(NULL, 0, ipaddr, NULL, -1);
// 	if (retval != expected) {
// 		PVLOG_ERROR("The IP addr '%s' is not mapped correctly expected '%f' but got '%f'!\n", ipaddr, expected, retval);
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


