/**
 * \file PVMappingFilterDateBase.cpp
 *
 * Copyright (C) Picviz Labs 2014
 */

#include <pvkernel/core/PVUnicodeString.h>
#include <pvkernel/core/PVDateTimeParser.h>

#include "PVMappingFilterDateBase.h"

#include <QStringList>

#include <unicode/calendar.h>
#include <unicode/ucal.h>

static int32_t cal_to_int(Calendar* cal, UCalendarDateFields sym, bool& success)
{
	UErrorCode err = U_ZERO_ERROR;
	int32_t m = cal->get(sym, err);

	success = U_SUCCESS(err);

	if (success) {
		return m;
	} else {
		return 0;
	}
}

Picviz::date_tls_parser::date_tls_parser() :
	_cal(nullptr)
{}

Picviz::date_tls_parser::~date_tls_parser()
{
	if (_cal) {
		delete _cal;
	}
	if (_parser) {
		delete _parser;
	}
}

void Picviz::date_tls_parser::init(QStringList const& time_format)
{
	if (_cal) {
		return;
	}
	UErrorCode err = U_ZERO_ERROR;
	_cal = Calendar::createInstance(err);
	_parser = new PVCore::PVDateTimeParser(time_format);
}

void Picviz::PVMappingFilterDateBase::init()
{
}

Picviz::PVMappingFilter::decimal_storage_type Picviz::date_mapping::process_utf8(const char* buf,
                                                                                 size_t size,
                                                                                 PVMappingFilter* m)
{
	QString stmp(QString::fromUtf8(buf, size));
	return process_utf16((const uint16_t*) stmp.constData(), stmp.size(), m);
}

Picviz::PVMappingFilter::decimal_storage_type Picviz::date_mapping::process_utf16(uint16_t const* buf,
                                                                                  size_t size,
                                                                                  PVMappingFilter* m)
{
	Picviz::PVMappingFilter::decimal_storage_type ret_ds;

	PVMappingFilterDateBase* mfdb = static_cast<PVMappingFilterDateBase*>(m);
	date_tls_parser& parser = mfdb->tls_parsers().local();
	parser.init(mfdb->get_time_format());
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

	ret_ds.storage_as_int() = cal_to_int(cal, mfdb->get_time_symbol(), success);

	if (!success) {
		ret_ds.storage_as_int() = 0;
		return ret_ds;
	}

	return ret_ds;
}

Picviz::PVMappingFilterDateBase::PVMappingFilterDateBase(PVCore::PVArgumentList const& args):
	PVPureMappingFilter<date_mapping>()
{
	INIT_FILTER(PVMappingFilterDateBase, args);
}

DEFAULT_ARGS_FILTER(Picviz::PVMappingFilterDateBase)
{
	PVCore::PVArgumentList args;
	return args;
}
