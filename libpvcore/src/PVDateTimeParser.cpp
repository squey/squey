#include <pvcore/PVDateTimeParser.h>
#include <QDate>

#include <boost/bind.hpp>

// No copy is made. The QString must remain valid as long as the UnicodeString object is !
static UnicodeString icuFromQStringAlias(const QString& src)
{
	return UnicodeString(false, (const UChar *)(src.constData()), src.size());
}

PVCore::PVDateTimeParser::PVDateTimeParser()
{
}

PVCore::PVDateTimeParser::PVDateTimeParser(const PVDateTimeParser& src)
{
	copy(src);
}

PVCore::PVDateTimeParser& PVCore::PVDateTimeParser::operator=(const PVDateTimeParser& src)
{
	if (&src != this) {
		copy(src);
	}
	return *this;
}

void PVCore::PVDateTimeParser::destroy_tf(TimeFormat* p)
{
	_alloc_tf.free((TimeFormat*) p);
}

void PVCore::PVDateTimeParser::destroy_tfe(TimeFormatEpoch* p)
{
	_alloc_tfe.free(p);
}

PVCore::PVDateTimeParser::PVDateTimeParser(QStringList const& time_format)
{
	// Compute the current year
	// If we are very unlucky, the year can change between this constructor and the computation of the struct tm's... !
	QString current_year = QDate::currentDate().toString("yyyy") + QString(" ");
	_current_year = UnicodeString((UChar*) current_year.data(), current_year.size());

	_time_format.reserve(time_format.size());
	QStringList::const_iterator it;
	for (it = time_format.begin(); it != time_format.end(); it++) {
		QString format_str(*it);
		bool is_epoch = (format_str.compare("epoch") == 0);
		if (is_epoch) {
			TimeFormatEpoch_p tf(_alloc_tfe.construct(), boost::bind(&PVDateTimeParser::destroy_tfe, this, _1));
			_time_format.push_back(tf);
		}
		else {
			bool prepend_year_to_value = !format_str.contains(QChar('y'));
			if (prepend_year_to_value) {
				PVLOG_DEBUG("This string does not contain a year, adding one\n");
				format_str.prepend("yyyy ");
			}
			TimeFormat_p tf(_alloc_tf.construct(format_str, prepend_year_to_value), boost::bind(&PVDateTimeParser::destroy_tf, this, _1));
			tf->current_year = _current_year;
			_time_format.push_back(tf);
		}
	}
}

void PVCore::PVDateTimeParser::copy(const PVDateTimeParser& src)
{
	list_time_format::const_iterator it;
	for (it = src._time_format.begin(); it != src._time_format.end(); it++) {
		// Use RTII to find out the real type of the TimeFormatInterface object.
		TimeFormatInterface* tfi = it->get();
		TimeFormat* tf = dynamic_cast<TimeFormat*>(tfi);
		if (tf == NULL) {
			TimeFormatEpoch_p ptfe(_alloc_tfe.construct(), boost::bind(&PVDateTimeParser::destroy_tfe, this, _1));
			_time_format.push_back(ptfe);
		}
		else {
			TimeFormat_p ptf(_alloc_tf.construct(*tf), boost::bind(&PVDateTimeParser::destroy_tf, this, _1));
			_time_format.push_back(ptf);
		}
	}

	_current_year = src._current_year;
	_last_match_time_format = *(_time_format.begin());
}

bool PVCore::PVDateTimeParser::mapping_time_to_cal(QString const& value, Calendar* cal)
{
	UnicodeString v = icuFromQStringAlias(value);
	if (_last_match_time_format) {
		if (_last_match_time_format->to_datetime(v, cal))
			return true;
	}

	list_time_format::iterator it;
	for (it = _time_format.begin(); it != _time_format.end(); it++) {
		TimeFormatInterface_p cur_tf = *it;
		if (cur_tf == _last_match_time_format) {
			continue;
		}
		if (cur_tf->to_datetime(v, cal)) {
			_last_match_time_format = cur_tf;
			return true;
		}
	}

	return false;
}

PVCore::PVDateTimeParser::TimeFormat::TimeFormat(QString const& time_format, bool prepend_year) :
	_err(U_ZERO_ERROR),
	time_format_(time_format),
	local_parser(icuFromQStringAlias(time_format), _err)
{
	prepend_year_value = prepend_year;
	create_parsers(time_format);

	// The "local" parser is the one using the system's local
	//local_parser = SimpleDateFormat(pattern, err);
	last_good_parser = &local_parser;
}

void PVCore::PVDateTimeParser::TimeFormat::create_parsers(QString const& time_format)
{
	int32_t nlocales;
	const Locale* list_locales = Locale::getAvailableLocales(nlocales);
	parsers.reserve(nlocales);
	UnicodeString pattern = icuFromQStringAlias(time_format);

	for (int il = 0; il < nlocales; il++) {
		const Locale &cur_loc = list_locales[il];
		UErrorCode err = U_ZERO_ERROR;
#ifdef WIN32
		// MSVC seems not to be able to take the good constructor for SimpleDateFormat...
		SimpleDateFormat *psdf = _alloc_df.malloc();
		new (psdf) SimpleDateFormat(pattern, Locale::createFromName(cur_loc.getName()), err);
#else
		SimpleDateFormat *psdf = _alloc_df.construct(pattern, Locale::createFromName(cur_loc.getName()), err);
#endif
		SimpleDateFormat_p sdf(psdf, boost::bind(&TimeFormat::destroy_sdf, this, _1));
		if (U_SUCCESS(err)) {
			parsers.push_back(sdf);
		}
		else {
			PVLOG_WARN("Unable to create parser for locale %s.\n", cur_loc.getName());
		}
	}
}

PVCore::PVDateTimeParser::TimeFormat::TimeFormat(const TimeFormat& src):
	local_parser(icuFromQStringAlias(src.time_format_), _err)
{
	copy(src);
}

PVCore::PVDateTimeParser::TimeFormat::~TimeFormat()
{
}

PVCore::PVDateTimeParser::TimeFormat& PVCore::PVDateTimeParser::TimeFormat::operator=(const TimeFormat& src)
{
	if (&src != this) {
		copy(src);
	}
	return *this;
}

void PVCore::PVDateTimeParser::TimeFormat::destroy_sdf(SimpleDateFormat* p)
{
	_alloc_df.destroy(p);
}

void PVCore::PVDateTimeParser::TimeFormat::copy(TimeFormat const& src)
{
	prepend_year_value = src.prepend_year_value;
	local_parser = src.local_parser;
	current_year = src.current_year;
	// Set the last_good_parser to the "local parser" of this object
	last_good_parser = &local_parser;
	time_format_ = src.time_format_;
	create_parsers(time_format_);
}


bool PVCore::PVDateTimeParser::TimeFormat::to_datetime(UnicodeString const& value, Calendar* cal)
{
	UErrorCode err = U_ZERO_ERROR;
	cal->setTime(0, err);
	cal->setTimeZone(*TimeZone::getGMT());
	UnicodeString value_(value);

	if (prepend_year_value) {
		value_ = current_year;
		value_ += value;
	}

	ParsePosition pos(0);
	last_good_parser->parse(value, *cal, pos);

	if (pos.getErrorIndex() == -1) {
		return true;
	}

	std::vector<SimpleDateFormat_p>::iterator it;
	for (it = parsers.begin(); it != parsers.end(); it++) {
		SimpleDateFormat *cur_parser = it->get();
		if (cur_parser->getSmpFmtLocale() == last_good_parser->getSmpFmtLocale()) {
			continue;
		}
		ParsePosition pos_(0);
		cur_parser->parse(value_, *cal, pos_);
		if (pos_.getErrorIndex() == -1) {
			last_good_parser = cur_parser;
			return true;
		}
	}

	return false;
}

bool PVCore::PVDateTimeParser::TimeFormatEpoch::to_datetime(UnicodeString const& value, Calendar* cal)
{
	UErrorCode err = U_ZERO_ERROR;
	cal->setTime(0, err);
	cal->setTimeZone(*TimeZone::getGMT());
	QString tmp = QString::fromRawData(reinterpret_cast<const QChar *>(value.getBuffer()), value.length());
	bool ok = false;
	UDate date = tmp.toLongLong(&ok);
	if (!ok)
		return false;
	err = U_ZERO_ERROR;
	cal->setTime(date, err);

	return U_SUCCESS(err);
}
