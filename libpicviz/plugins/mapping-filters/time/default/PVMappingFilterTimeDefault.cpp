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
	args[PVCore::PVArgumentKey("time-format", "Format of the\ntime strings :")].setValue(tf);
	return args;
}

float* Picviz::PVMappingFilterTimeDefault::operator()(PVRush::PVNraw::nraw_table_line const& values)
{
	assert(_dest);
	assert(values.size() >= _dest_size);
	assert(_format);

	//PVCore::PVDateTimeParser dtpars(_format->time_format[_cur_col+1]);
	//UErrorCode err = U_ZERO_ERROR;
	//Calendar* cal = Calendar::createInstance(err);
	// Create calender and parsers objects that will be used by our threads
	const int max_threads = omp_get_max_threads();
	Calendar** cals = new Calendar*[max_threads];
	PVCore::PVDateTimeParser **dtparsers = new PVCore::PVDateTimeParser*[max_threads];
	tbb::tick_count start_alloc = tbb::tick_count::now();
	PVRush::PVAxisFormat const& axis = _format->get_axes().at(_cur_col);
	QStringList time_format(axis.get_args_mapping()["time-format"].value<PVCore::PVTimeFormatType>());
	for (int i = 0; i < max_threads; i++) {
		UErrorCode err = U_ZERO_ERROR;
		cals[i] = Calendar::createInstance(err);
		//dtparsers[i] = new PVCore::PVDateTimeParser(_format->time_format[_cur_col+1]);
		dtparsers[i] = new PVCore::PVDateTimeParser(time_format);
	}
	tbb::tick_count end_alloc = tbb::tick_count::now();
	PVLOG_DEBUG("(PVMappingFilterTimeDefault::operator()) object creations took %0.4fs.\n", (end_alloc-start_alloc).seconds());

	int64_t size = _dest_size;
	// TODO: compare TBB and OpenMP here !!
#pragma omp parallel for schedule(dynamic, 1000)
	for (int64_t i = 0; i < size; i++) {
		int thread_num = omp_get_thread_num();
		Calendar* cal = cals[thread_num];
		PVCore::PVDateTimeParser &dtpars = *(dtparsers[thread_num]);
		if (values[i].isEmpty()) {
			_dest[i] = 0;
			continue;
		}
		bool ret = dtpars.mapping_time_to_cal(values[i], cal);
		if (!ret) {
#pragma omp critical
			{
				PVLOG_WARN("(time-mapping) unable to map time string %s. Returns 0 !\n", qPrintable(values[i]));
			}
			_dest[i] = 0;
			continue;
		}

		bool success;
		_dest[i] = cal_to_float(cal, success);
		if (!success) {
#pragma omp critical
			{
				PVLOG_WARN("(time-mapping) unable to map time string %s: one field is missing. Returns 0 !\n", qPrintable(values[i]));
			}
			_dest[i] = 0;
			continue;
		}
	}

	// Frees the calendar objects
	for (int i = 0; i < max_threads; i++) {
		delete cals[i];
		delete dtparsers[i];
	}
	delete [] cals;
	delete [] dtparsers;

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
