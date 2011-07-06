#include <stdio.h>

#include <picviz/general.h>
#include <picviz/function.h>

#include <picviz/PVMapping.h>

LibCPPExport float picviz_mapping_exec(const Picviz::PVMapping_p mapping, PVCol index, QString &value, void *userdata, bool is_first)
{
/* 16105 is the value corresponding to the arbitrary string:
 * "The competent programmer is fully aware of the limited size of his own skull. He therefore approaches his task with full humility, and 
 * avoids clever tricks like the plague."
 */
#define STRING_MAX_YVAL 16105

struct string_ascii_val_t {
	float max;
};

	float factor = 0;
	float retval;
	struct string_ascii_val_t string_ascii_val;

	while (*value++) {
		char c = *value;
		factor += c;
	}

	if ( is_first ) {
		string_ascii_val.max = factor;
		PICVIZ_USERDATA(userdata, struct string_ascii_val_t) = string_ascii_val;
	} else {
		string_ascii_val = PICVIZ_USERDATA(userdata, struct string_ascii_val_t);
		if ( factor > string_ascii_val.max ) {
			string_ascii_val.max = factor;
			PICVIZ_USERDATA(userdata, struct string_ascii_val_t) = string_ascii_val;
		}
	}

	if ( string_ascii_val.max > STRING_MAX_YVAL ) {
		retval = factor / string_ascii_val.max;
	} else {
		retval = factor / STRING_MAX_YVAL;
	}

	return retval;
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

