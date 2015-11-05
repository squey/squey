/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVMAPPINGFILTERDATEBASE_H
#define PVFILTER_PVMAPPINGFILTERDATEBASE_H

#include <pvkernel/core/general.h>
#include <inendi/PVPureMappingFilter.h>

#include <unicode/calendar.h>

#include <tbb/enumerable_thread_specific.h>

namespace PVCore
{

class PVDateTimeParser;

}

namespace Inendi
{

struct date_tls_parser
{
	date_tls_parser();
	~date_tls_parser();

	void init(QStringList const& time_format);

	Calendar* cal() { return _cal; }
	PVCore::PVDateTimeParser& parser() { return *_parser; }

private:
	Calendar*                 _cal;
	PVCore::PVDateTimeParser* _parser;
};

struct date_mapping
{
	static Inendi::PVMappingFilter::decimal_storage_type process_utf8(const char* buf,
	                                                                  size_t size,
	                                                                  PVMappingFilter* m);
	static Inendi::PVMappingFilter::decimal_storage_type process_utf16(uint16_t const* buf,
	                                                                   size_t size,
	                                                                   PVMappingFilter* m);
};

class PVMappingFilterDateBase: public PVPureMappingFilter<date_mapping>
{
	friend class date_mapping;

public:
	PVMappingFilterDateBase(PVCore::PVArgumentList const& args = PVMappingFilterDateBase::default_args());

public:
	QString get_human_name() const override { return QString("Base"); }
	PVCore::DecimalType get_decimal_type() const override { return PVCore::IntegerType; }
	void init() override;

protected:
	inline void set_time_format(const QStringList& sl) { _time_format = sl; }
	inline QStringList const& get_time_format() const { return _time_format; }

	inline void set_time_symbol(UCalendarDateFields sym) { _time_sym = sym; }
	inline UCalendarDateFields get_time_symbol() { return _time_sym; }

protected:
	inline tbb::enumerable_thread_specific<date_tls_parser>& tls_parsers() { return _tls_parsers; }

private:
	QStringList _time_format;
	UCalendarDateFields _time_sym;
	tbb::enumerable_thread_specific<date_tls_parser> _tls_parsers;

	CLASS_FILTER(PVMappingFilterDateBase)
};

}

#endif // PVFILTER_PVMAPPINGFILTERDATEBASE_H
