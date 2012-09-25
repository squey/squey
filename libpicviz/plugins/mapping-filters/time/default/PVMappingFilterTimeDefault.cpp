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
	PVPureMappingFilter<time_mapping>()
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

Picviz::tls_parser::tls_parser():
	_cal(nullptr)
{
}

Picviz::tls_parser::~tls_parser()
{
	if (_cal) {
		delete _cal;
		delete _parser;
	}
}

void Picviz::tls_parser::init(QStringList const& time_format)
{
	if (_cal) {
		return;
	}
	UErrorCode err = U_ZERO_ERROR;
	_cal = Calendar::createInstance(err);
	_parser = new PVCore::PVDateTimeParser(time_format);
}

void Picviz::PVMappingFilterTimeDefault::init()
{
	_time_format = _args["time-format"].value<PVCore::PVTimeFormatType>();
}

Picviz::PVMappingFilter::decimal_storage_type Picviz::time_mapping::process_utf8(const char* buf, size_t size, PVMappingFilter* m)
{
	QString stmp(QString::fromUtf8(buf, size));
	return process_utf16((const uint16_t*) stmp.constData(), stmp.size(), m);
}

Picviz::PVMappingFilter::decimal_storage_type Picviz::time_mapping::process_utf16(uint16_t const* buf, size_t size, PVMappingFilter* m)
{
	Picviz::PVMappingFilter::decimal_storage_type ret_ds;
	tls_parser& parser = static_cast<PVMappingFilterTimeDefault*>(m)->tls_parsers().local();
	parser.init(static_cast<PVMappingFilterTimeDefault*>(m)->time_format());
	Calendar* cal = parser.cal();
	PVCore::PVDateTimeParser &dtpars = parser.parser();
	PVCore::PVUnicodeString16 const v(buf, size);
	if (v.size() == 0) {
		ret_ds.storage_as_int() = 0;
		return ret_ds;
	}
	bool ret = dtpars.mapping_time_to_cal(v, cal);
	if (!ret) {
		ret_ds.storage_as_int() = 0;
		return ret_ds;
	}

	bool success;
	ret_ds.storage_as_int() = static_cast<PVMappingFilterTimeDefault*>(m)->cal_to_int(cal, success);
	if (!success) {
		ret_ds.storage_as_int() = 0;
		return ret_ds;
	}

	return ret_ds;
}

int32_t Picviz::PVMappingFilterTimeDefault::cal_to_int(Calendar* cal, bool& success)
{
	UErrorCode err = U_ZERO_ERROR;
	int32_t ret = (int32_t) cal->getTime(err);
	success = U_SUCCESS(err);
	return ret;
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterTimeDefault)
