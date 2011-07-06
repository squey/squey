#ifndef PVCORE_PVDATETIMEPARSER_H
#define PVCORE_PVDATETIMEPARSER_H

#include <unicode/smpdtfmt.h>
#include <unicode/locid.h>
#include <unicode/unistr.h>

#include <QStringList>
#include <QString>

#include <vector>

#include <pvcore/general.h>

#include <boost/pool/singleton_pool.hpp>
#include <boost/shared_ptr.hpp>

namespace PVCore {

class LibExport PVDateTimeParser {
public:
	PVDateTimeParser();
	PVDateTimeParser(QStringList const& time_format);

public:
	bool mapping_time_to_cal(QString const& value_, Calendar* cal);

private:
	void copy(const PVDateTimeParser& src);

protected:
	struct TimeFormatInterface {
		virtual ~TimeFormatInterface() {}
		virtual bool to_datetime(UnicodeString const& value, Calendar* cal) = 0;
	};

	class TimeFormat : public TimeFormatInterface {
	private:
		UErrorCode _err;
		bool own_parsers;
		bool is_epoch;
	public:
		TimeFormat(QString const& time_format, bool prepend_year);
		TimeFormat(const TimeFormat&);
		~TimeFormat();
		TimeFormat& operator=(const TimeFormat& src);
		bool to_datetime(UnicodeString const& value, Calendar* cal);

		bool prepend_year_value;
		// One object per locale
		std::vector<SimpleDateFormat*> parsers;
		SimpleDateFormat local_parser;
		SimpleDateFormat* last_good_parser;
		UnicodeString current_year;
	};

	struct TimeFormatEpoch : public TimeFormatInterface {
		bool to_datetime(UnicodeString const& value, Calendar* cal);
	};

	typedef boost::shared_ptr<TimeFormat> TimeFormat_p;
	typedef boost::shared_ptr<TimeFormatEpoch> TimeFormatEpoch_p;
	typedef boost::shared_ptr<TimeFormatInterface> TimeFormatInterface_p;

	typedef boost::singleton_pool<TimeFormat, sizeof(TimeFormat)> alloc_tf_t;
	typedef boost::singleton_pool<TimeFormatEpoch, sizeof(TimeFormatEpoch)> alloc_tfe_t;

private:
	static void destroy_tf(TimeFormatInterface* p);

protected:
	typedef std::vector<TimeFormatInterface_p> list_time_format;
	list_time_format _time_format;

	UnicodeString _current_year;
	TimeFormatInterface_p _last_match_time_format;
	bool own_tfs;
};

}

#endif
