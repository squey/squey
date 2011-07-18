#include "PVMappingFilterTimeWeek.h"

#include <unicode/ucal.h>

float Picviz::PVMappingFilterTimeWeek::cal_to_float(Calendar* cal, bool &success)
{
	UErrorCode err = U_ZERO_ERROR;
	int32_t sec = cal->get(UCAL_SECOND, err);
	int32_t min = cal->get(UCAL_MINUTE, err);
	int32_t hour = cal->get(UCAL_HOUR_OF_DAY, err);
	int32_t day = cal->get(UCAL_DAY_OF_WEEK, err) - 1;

	success = U_SUCCESS(err);

	if (success) {
		return (float) (sec +  (min * 60) + (hour * 60 * 60) + day*24*3600);
	}
	return 0;
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterTimeWeek)
