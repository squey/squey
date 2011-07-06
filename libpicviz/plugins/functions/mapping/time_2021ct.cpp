/* time mode 20-21st century = from 1900 to 2100 */

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


/* #include <win32-strptime.h> */

#include <picviz/general.h>
#include <picviz/debug.h>
#include <picviz/format.h>
#include <picviz/function.h>
#include <picviz/limits.h>
#include <picviz/utils.h>

#include <picviz/gnulib/strptime.h>

#include <picviz/PVMapping.h>

/*
 * Function mapping for values.
 * each mapping function has the following prototype:
 * float picviz_mapping_function_TYPE_MODE(char *value, void *userdata, int is_first)
  */
LibCPPExport float picviz_mapping_exec(const Picviz::PVMapping_p mapping, PVCol index, QString &value, void *userdata, bool is_first)
{
	picviz_format_t *format;

	time_t t;
	struct tm tm;
	char *time_format;
	char *retval;
	float factor = 0;
	int h_m_s_pos;
	int month_pos;

	int has_year;
	char *new_format;
	char *new_value;

	char current_year[5];
	int int_year;
	char current_month[3];
	int int_month;
	char current_day[3];
	int int_day;

	if (!strcmp(value,"")) return 0;

	current_year[0] = value[0];
	current_year[1] = value[1];
	current_year[2] = value[2];
	current_year[3] = value[3];
	current_year[4] = '\0';
	int_year = atoi(current_year);

	current_month[0] = value[5];
	current_month[1] = value[6];
	current_month[2] = '\0';
	int_month = atoi(current_month);

	current_day[0] = value[8];
	current_day[1] = value[9];
	current_day[2] = '\0';
	int_day = atoi(current_day);

	factor = (float) (int_year - 1900) * 12 + int_month;

	return factor;
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
