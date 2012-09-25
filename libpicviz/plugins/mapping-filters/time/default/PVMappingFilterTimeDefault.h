/**
 * \file PVMappingFilterTimeDefault.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVMAPPINGFILTERTIMEDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERTIMEDEFAULT_H

#include <pvkernel/core/general.h>
#include <picviz/PVPureMappingFilter.h>

#include <unicode/calendar.h>

#include <tbb/enumerable_thread_specific.h>

namespace PVCore {
class PVDateTimeParser;
}

namespace Picviz {

struct time_mapping
{
	static Picviz::PVMappingFilter::decimal_storage_type process_utf8(const char* buf, size_t size, PVMappingFilter* m);
	static Picviz::PVMappingFilter::decimal_storage_type process_utf16(uint16_t const* buf, size_t size, PVMappingFilter* m);
};

struct tls_parser
{
	tls_parser();
	~tls_parser();

	void init(QStringList const& time_format);

	Calendar* cal() { return _cal; }
	PVCore::PVDateTimeParser& parser() { return *_parser; }
private:
	Calendar* _cal;
	PVCore::PVDateTimeParser* _parser;
};

class PVMappingFilterTimeDefault: public PVPureMappingFilter<time_mapping>
{
	friend class time_mapping;
public:
	PVMappingFilterTimeDefault(PVCore::PVArgumentList const& args = PVMappingFilterTimeDefault::default_args());

public:
	QString get_human_name() const override { return QString("Default"); }
	PVCore::DecimalType get_decimal_type() const override { return PVCore::UnsignedIntegerType; }
	void init() override;

protected:
	inline QStringList const& time_format() const { return _time_format; }
	inline tbb::enumerable_thread_specific<tls_parser>& tls_parsers() { return _tls_parsers; }

protected:
	virtual int32_t cal_to_int(Calendar* cal, bool& success);

private:
	QStringList _time_format;
	tbb::enumerable_thread_specific<tls_parser> _tls_parsers;

	CLASS_FILTER(PVMappingFilterTimeDefault)
};

}

#endif
