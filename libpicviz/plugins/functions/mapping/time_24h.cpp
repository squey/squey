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

#include <QDate>
#include <QDateTime>
#include <QString>

#include <pvcore/debug.h>
#include <pvcore/environ.h>

#include <pvrush/PVFormat.h>

#include <picviz/general.h>

#include <picviz/PVMapping.h>

#include "time/core.h"

using namespace PVCore;

/*
 * Function mapping for values.
 * each mapping function has the following prototype:
 * float picviz_mapping_function_TYPE_MODE(char *value, void *userdata, int is_first)
  */
LibCPPExport float picviz_mapping_exec(const Picviz::PVMapping_p mapping, PVCol index, QString &value, void *userdata, bool is_first)
{
	PVRush::PVFormat *format;

	struct tm tm_buf; // we need it so that this mapping function can be multi-threaded
	struct tm *tm;
	QStringList time_format;

	char *retval;
	float factor = 0;

	int axis_id;
	int count;

	QString qstr_value(value);

	format = mapping->get_format();
	axis_id = index + 1;

	time_format = format->time_format[axis_id];

	tm = picviz_mapping_time_to_tm(time_format, qstr_value, &tm_buf);
	if (!tm) {
		return 0;
	}

	factor = (float)tm->tm_sec +  (tm->tm_min * 60) + (tm->tm_hour * 60 * 60);

	PVLOG_HEAVYDEBUG("Mapping time 24h: retval = '%f'\n", factor);

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

