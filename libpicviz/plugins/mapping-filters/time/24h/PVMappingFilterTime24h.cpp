/**
 * \file PVMappingFilterTime24h.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVMappingFilterTime24h.h"

#include <unicode/ucal.h>

float Picviz::PVMappingFilterTime24h::cal_to_float(Calendar* cal, bool &success)
{
	UErrorCode err = U_ZERO_ERROR;
	int32_t sec = cal->get(UCAL_SECOND, err);
	int32_t min = cal->get(UCAL_MINUTE, err);
	int32_t hour = cal->get(UCAL_HOUR_OF_DAY, err);

	success = U_SUCCESS(err);

	if (success) {
		return (float) (sec +  (min * 60) + (hour * 60 * 60));
	}
	return 0;
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterTime24h)
