/* Plotting float default */

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
	int week_pos;

	int has_year;
	char *new_format;
	char *new_value;

	char current_year[5];

	int axis_id;


	format = mapping->PICVIZ_OBJECT_SOURCE(parent)->format;
	axis_id = index + 1;
	time_format = picviz_format_get_time_format(format, axis_id);

	has_year = picviz_string_count_char(time_format, 'Y');
	if (has_year) {
		retval = picviz_strptime(value, time_format, &tm);
	} else {
		t = time(NULL);
		strftime(current_year, 5, "%Y", localtime(&t));
		new_value = malloc(strlen(value) + 6);
		sprintf(new_value, "%s %s", value, current_year);
		new_format = malloc(strlen(time_format) + 4);
		sprintf(new_format, "%s " ESCAPE_PERCENT "Y", time_format);
		retval = picviz_strptime(new_value, new_format, &tm);
		free(new_value);
		free(new_format);
	}

	if (retval) {
		h_m_s_pos = tm.tm_sec +  (tm.tm_min * 60) + (tm.tm_hour * 60 * 60);
		week_pos = tm.tm_wday * PICVIZ_TIME_24H_MAX;
		factor = (float)week_pos + h_m_s_pos;
	}


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

