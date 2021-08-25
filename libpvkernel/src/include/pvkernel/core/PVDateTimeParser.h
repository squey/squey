/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVCORE_PVDATETIMEPARSER_H
#define PVCORE_PVDATETIMEPARSER_H

#include <unicode/smpdtfmt.h>
#include <unicode/locid.h>
#include <unicode/unistr.h>

#include <QStringList>
#include <QString>

#include <vector>

#include <boost/pool/object_pool.hpp>

using namespace icu_67;

namespace PVCore
{

class PVDateTimeParser
{
  public:
	PVDateTimeParser();
	explicit PVDateTimeParser(QStringList const& time_format);
	PVDateTimeParser(const PVDateTimeParser& src);
	~PVDateTimeParser();

	PVDateTimeParser& operator=(const PVDateTimeParser& src);

  public:
	inline bool mapping_time_to_cal(QString const& value, Calendar* cal)
	{
		return mapping_time_to_cal(icuFromQStringAlias(value), cal);
	}
	bool mapping_time_to_cal(UnicodeString const& v, Calendar* cal);
	QStringList const& original_time_formats() const { return _org_time_format; }

  private:
	void copy(const PVDateTimeParser& src);

  protected:
	struct TimeFormatInterface {
		virtual ~TimeFormatInterface() = default;
		virtual bool to_datetime(UnicodeString const& value, Calendar* cal) = 0;
	};

	class TimeFormat : public TimeFormatInterface
	{
		typedef SimpleDateFormat* SimpleDateFormat_p;
		// typedef std::shared_ptr<SimpleDateFormat> SimpleDateFormat_p;
	  private:
		UErrorCode _err;
		QString time_format_;
		SimpleDateFormat* _parsers;
		size_t _nparsers;

	  public:
		TimeFormat(QString const& time_format, bool prepend_year);
		TimeFormat(const TimeFormat&);
		~TimeFormat() override;
		TimeFormat& operator=(const TimeFormat& src);
		bool to_datetime(UnicodeString const& value, Calendar* cal) override;

		bool prepend_year_value;
		// One object per locale
		// std::vector<SimpleDateFormat_p> parsers;
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
		bool to_datetime(UnicodeString const& value, Calendar* cal) override;
	};

	//	typedef std::shared_ptr<TimeFormat> TimeFormat_p;
	//	typedef std::shared_ptr<TimeFormatEpoch> TimeFormatEpoch_p;
	//	typedef std::shared_ptr<TimeFormatInterface> TimeFormatInterface_p;
	typedef TimeFormat* TimeFormat_p;
	typedef TimeFormatEpoch* TimeFormatEpoch_p;
	typedef TimeFormatInterface* TimeFormatInterface_p;

	// boost::object_pool<TimeFormat> _alloc_tf;
	// boost::object_pool<TimeFormatEpoch> _alloc_tfe;

  private:
	static UnicodeString icuFromQStringAlias(const QString& src);

  protected:
	typedef std::vector<TimeFormatInterface_p> list_time_format;
	list_time_format _time_format;

	UnicodeString _current_year;
	TimeFormatInterface_p _last_match_time_format;

	QStringList _org_time_format;
};
} // namespace PVCore

#endif
