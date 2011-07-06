#include "PVMappingFilterTimeDefault.h"
#include <pvrush/PVFormat.h>
#include <pvcore/PVDateTimeParser.h>

#include <QStringList>
#include <omp.h>

#include <unicode/calendar.h>
#include <unicode/ucal.h>

#ifdef WIN32
#include <pvcore/win32-vs2008-stdint.h>
#else
#include <stdint.h>
#endif

// Ok, we can't use this with gcc... That's an open bug from 2006 !!
// See http://gcc.gnu.org/bugzilla/show_bug.cgi?id=27557
// QStringList ltmp;
// PVCore::PVDateTimeParser dtpars(ltmp);
// #pragma omp threadprivate(dtpars)

float* Picviz::PVMappingFilterTimeDefault::operator()(PVRush::PVNraw::nraw_table_line const& values)
{
	assert(_dest);
	assert(values.size() >= _dest_size);
	assert(_format);

	PVCore::PVDateTimeParser dtpars_org(_format->time_format[_cur_col+1]);

	// TODO: vectorize this for RWBITS bits register width by loading N=RWBITS/sizeof(float) time structures
	// and compute from dest[i] to dest[i+N] using vectorized operations !
	// It would use SSE in a first time, and AVX when this will be supported !
	
	// Create calender and parsers objects that will be used by our threads
	const int max_threads = omp_get_max_threads();
#ifdef WIN32
	Calendar** cals = (Calendar**) malloc(max_threads*sizeof(Calendar*));
	PVCore::PVDateTimeParser *dtparsers = (PVCore::PVDateTimeParser*) malloc(max_threads*sizeof(PVCore::PVDateTimeParser));
#else
	Calendar* cals[max_threads];
	PVCore::PVDateTimeParser dtparsers[max_threads];
#endif
	for (int i = 0; i < max_threads; i++) {
		UErrorCode err = U_ZERO_ERROR;
		cals[i] = Calendar::createInstance(err);
#ifdef WIN32
		new(&dtparsers[i]) PVCore::PVDateTimeParser();
#endif
		dtparsers[i] = dtpars_org;
	}

	int64_t size = _dest_size;
	// TODO: compare TBB and OpenMP here !!
#pragma omp parallel for
	for (int64_t i = 0; i < size; i++) {
		int thread_num = omp_get_thread_num();
		Calendar* cal = cals[thread_num];
		PVCore::PVDateTimeParser &dtpars = dtparsers[thread_num];
		bool ret = dtpars.mapping_time_to_cal(values[i], cal);
		if (!ret) {
#pragma omp critical
			{
				PVLOG_WARN("(time-24h mapping) unable to map time string %s. Returns 0 !\n", qPrintable(values[i]));
			}
			_dest[i] = 0;
			continue;
		}

		bool success;
		_dest[i] = cal_to_float(cal, success);
		if (!success) {
#pragma omp critical
			{
				PVLOG_WARN("(time-24h mapping) unable to map time string %s: one field is missing. Returns 0 !\n", qPrintable(values[i]));
			}
			_dest[i] = 0;
			continue;
		}
	}

	// Frees the calendar objects
	for (int i = 0; i < max_threads; i++) {
		delete cals[i];
	}
#ifdef WIN32
	free(cals);
	free(dtparsers);
#endif

	return _dest;
}

float Picviz::PVMappingFilterTimeDefault::cal_to_float(Calendar* cal, bool& success)
{
	UErrorCode err = U_ZERO_ERROR;
	float ret = cal->getTime(err);
	success = U_SUCCESS(err);
	return ret;
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterTimeDefault)
