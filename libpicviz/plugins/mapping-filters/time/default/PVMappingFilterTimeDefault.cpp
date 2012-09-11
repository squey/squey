/**
 * \file PVMappingFilterTimeDefault.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVUnicodeString.h>
#include "PVMappingFilterTimeDefault.h"
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/core/PVDateTimeParser.h>
#include <pvkernel/core/PVTimeFormatType.h>
#include <pvkernel/core/stdint.h>

#include <QStringList>
#include <omp.h>

#include <unicode/calendar.h>
#include <unicode/ucal.h>

#include <tbb/tick_count.h>

// Ok, we can't use this with gcc... That's an open bug from 2006 !!
// See http://gcc.gnu.org/bugzilla/show_bug.cgi?id=27557
// QStringList ltmp;
// PVCore::PVDateTimeParser dtpars(ltmp);
// #pragma omp threadprivate(dtpars)

Picviz::PVMappingFilterTimeDefault::PVMappingFilterTimeDefault(PVCore::PVArgumentList const& args):
	PVMappingFilter()
{
	INIT_FILTER(PVMappingFilterTimeDefault, args);
}

DEFAULT_ARGS_FILTER(Picviz::PVMappingFilterTimeDefault)
{
	PVCore::PVArgumentList args;
	PVCore::PVTimeFormatType tf;
	args[PVCore::PVArgumentKey("time-format", "Time strings formats")].setValue(tf);
	return args;
}

Picviz::PVMappingFilter::decimal_storage_type* Picviz::PVMappingFilterTimeDefault::operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values)
{
	assert(_dest);
	assert(values.size() >= _dest_size);

	//PVCore::PVDateTimeParser dtpars(_format->time_format[_cur_col+1]);
	//UErrorCode err = U_ZERO_ERROR;
	//Calendar* cal = Calendar::createInstance(err);
	// Create calender and parsers objects that will be used by our threads
	const int max_threads = omp_get_max_threads();
	Calendar* cals[max_threads];
	PVCore::PVDateTimeParser *dtparsers[max_threads];

	// Get space on the stack for PVDateTimeParser objects
	char* buf_parsers = (char*) alloca(sizeof(PVCore::PVDateTimeParser)*max_threads);

	tbb::tick_count start_alloc = tbb::tick_count::now();
	QStringList time_format(_args["time-format"].value<PVCore::PVTimeFormatType>());
	for (int i = 0; i < max_threads; i++) {
		UErrorCode err = U_ZERO_ERROR;
		cals[i] = Calendar::createInstance(err);
		//dtparsers[i] = new PVCore::PVDateTimeParser(_format->time_format[_cur_col+1]);
		PVCore::PVDateTimeParser *pstack = (PVCore::PVDateTimeParser*) &buf_parsers[i*sizeof(PVCore::PVDateTimeParser)];
		new (pstack) PVCore::PVDateTimeParser(time_format);
		dtparsers[i] = pstack;
	}
	tbb::tick_count end_alloc = tbb::tick_count::now();
	PVLOG_DEBUG("(PVMappingFilterTimeDefault::operator()) object creations took %0.4fs.\n", (end_alloc-start_alloc).seconds());

	int64_t size = _dest_size;
	// TODO: compare TBB and OpenMP here !!
#pragma omp parallel for
	for (int64_t i = 0; i < size; i++) {
		int thread_num = omp_get_thread_num();
		Calendar* cal = cals[thread_num];
		PVCore::PVDateTimeParser &dtpars = *(dtparsers[thread_num]);
		PVCore::PVUnicodeString const v(values[i]);
		if (v.size() == 0) {
			_dest[i].storage_as_int() = 0;
			continue;
		}
		bool ret = dtpars.mapping_time_to_cal(v, cal);
		if (!ret) {
			/*
#pragma omp critical
			{
				PVLOG_WARN("(time-mapping) unable to map time string %s. Returns 0 !\n", qPrintable(v));
			}
			*/
			_dest[i].storage_as_int() = 0;
			continue;
		}

		bool success;
		_dest[i].storage_as_int() = cal_to_int(cal, success);
		if (!success) {
			/*
#pragma omp critical
			{
				PVLOG_WARN("(time-mapping) unable to map time string %s: one field is missing. Returns 0 !\n", qPrintable(v));
			}
			*/
			_dest[i].storage_as_int() = 0;
			continue;
		}
	}

	start_alloc = tbb::tick_count::now();
	// Frees the calendar objects
	for (int i = 0; i < max_threads; i++) {
		delete cals[i];
		//delete dtparsers[i];
	}

	end_alloc = tbb::tick_count::now();
	PVLOG_DEBUG("(PVMappingFilterTimeDefault::operator()) object destruction took %0.4fs.\n", (end_alloc-start_alloc).seconds());

	return _dest;
}

int32_t Picviz::PVMappingFilterTimeDefault::cal_to_int(Calendar* cal, bool& success)
{
	UErrorCode err = U_ZERO_ERROR;
	int32_t ret = (int32_t) cal->getTime(err);
	success = U_SUCCESS(err);
	return ret;
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterTimeDefault)
