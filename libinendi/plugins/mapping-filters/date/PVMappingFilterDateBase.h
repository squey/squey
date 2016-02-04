/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVMAPPINGFILTERDATEBASE_H
#define PVFILTER_PVMAPPINGFILTERDATEBASE_H

#include <pvkernel/core/general.h>
#include <inendi/PVMappingFilter.h>

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
	// FIXME conversion from utf8 to utf16 is useless.
	static Inendi::PVMappingFilter::decimal_storage_type process_utf8(const char* buf,
	                                                                  size_t size,
	                                                                  PVMappingFilter* m);
	static Inendi::PVMappingFilter::decimal_storage_type process_utf16(uint16_t const* buf,
	                                                                   size_t size,
	                                                                   PVMappingFilter* m);
};

class PVMappingFilterDateBase: public PVMappingFilter
{
	friend class date_mapping;

public:
	PVMappingFilterDateBase(PVCore::PVArgumentList const& args = PVMappingFilterDateBase::default_args());

public:
	decimal_storage_type* operator()(PVCol const col, PVRush::PVNraw const& nraw)
	{
		auto array = nraw.collection().column(col);
		for(size_t row=0; row< array.size(); row++) {
			std::string content = array.at(row);
			this->_dest[row] = date_mapping::process_utf8(content.c_str(), content.size(), this);
		}

		return this->_dest;
	}

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
