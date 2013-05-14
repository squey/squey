/**
 * \file PVDateTimeParser.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PVDATETIMEPARSER_H
#define PVCORE_PVDATETIMEPARSER_H

#include <pvkernel/core/PVUnicodeString16.h>

#include <unicode/smpdtfmt.h>
#include <unicode/locid.h>
#include <unicode/unistr.h>

#include <QStringList>
#include <QString>

#include <vector>

#include <pvkernel/core/general.h>

#include <boost/pool/object_pool.hpp>
#include <boost/shared_ptr.hpp>

namespace PVCore {

class LibKernelDecl PVDateTimeParser {
public:
	PVDateTimeParser();
	PVDateTimeParser(QStringList const& time_format);
	PVDateTimeParser(const PVDateTimeParser& src);
	~PVDateTimeParser();

	PVDateTimeParser& operator=(const PVDateTimeParser& src);

public:
	inline bool mapping_time_to_cal(QString const& value, Calendar* cal)
	{
		return mapping_time_to_cal(icuFromQStringAlias(value), cal);
	}
	inline bool mapping_time_to_cal(PVCore::PVUnicodeString16 const& src, Calendar* cal)
	{
		return mapping_time_to_cal(UnicodeString(false, (const UChar *)(src.buffer()), src.size()), cal);
	}
	bool mapping_time_to_cal(UnicodeString const& v, Calendar* cal);
	QStringList const& original_time_formats() const { return _org_time_format; }

private:
	void copy(const PVDateTimeParser& src);

protected:
	struct TimeFormatInterface {
		virtual ~TimeFormatInterface() {}
		virtual bool to_datetime(UnicodeString const& value, Calendar* cal) = 0;
	};

	class TimeFormat : public TimeFormatInterface {
		typedef SimpleDateFormat* SimpleDateFormat_p;
		//typedef boost::shared_ptr<SimpleDateFormat> SimpleDateFormat_p;
	private:
		UErrorCode _err;
		bool is_epoch;
		QString time_format_;
		SimpleDateFormat* _parsers;
		size_t _nparsers;
	public:
		TimeFormat(QString const& time_format, bool prepend_year);
		TimeFormat(const TimeFormat&);
		~TimeFormat();
		TimeFormat& operator=(const TimeFormat& src);
		bool to_datetime(UnicodeString const& value, Calendar* cal);

		bool prepend_year_value;
		// One object per locale
		//std::vector<SimpleDateFormat_p> parsers;
		SimpleDateFormat local_parser;
		SimpleDateFormat* last_good_parser;
		UnicodeString current_year;

	private:
		void create_parsers(QString const& time_format);
		void copy(TimeFormat const& src);

	private:
		boost::object_pool<SimpleDateFormat> _alloc_df;
	};

	struct TimeFormatEpoch : public TimeFormatInterface {
		bool to_datetime(UnicodeString const& value, Calendar* cal);
	};

//	typedef boost::shared_ptr<TimeFormat> TimeFormat_p;
//	typedef boost::shared_ptr<TimeFormatEpoch> TimeFormatEpoch_p;
//	typedef boost::shared_ptr<TimeFormatInterface> TimeFormatInterface_p;
	typedef TimeFormat* TimeFormat_p;
	typedef TimeFormatEpoch* TimeFormatEpoch_p;
	typedef TimeFormatInterface* TimeFormatInterface_p;

	//boost::object_pool<TimeFormat> _alloc_tf;
	//boost::object_pool<TimeFormatEpoch> _alloc_tfe;

private:
	static UnicodeString icuFromQStringAlias(const QString& src);


protected:
	typedef std::vector<TimeFormatInterface_p> list_time_format;
	list_time_format _time_format;

	UnicodeString _current_year;
	TimeFormatInterface_p _last_match_time_format;

	QStringList _org_time_format;
};

}

#endif
