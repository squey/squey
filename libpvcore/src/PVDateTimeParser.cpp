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
	own_tfs = false;
}

void PVCore::PVDateTimeParser::destroy_tf(TimeFormatInterface* p)
{
	// Use RTII to find out the real type of 'p'
	TimeFormatEpoch* pe = dynamic_cast<TimeFormatEpoch*>(p);
	if (pe) {
		alloc_tfe_t::free(pe);
	}
	else {
		alloc_tf_t::free((TimeFormat*) p);
	}
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
			TimeFormatEpoch* ptf = (TimeFormatEpoch*) alloc_tfe_t::malloc();
			new (ptf) TimeFormatEpoch();
			TimeFormatEpoch_p tf(ptf, &PVDateTimeParser::destroy_tf);
			_time_format.push_back(tf);
		}
		else {
			bool prepend_year_to_value = !format_str.contains(QChar('y'));
			if (prepend_year_to_value) {
				PVLOG_DEBUG("This string does not contain a year, adding one\n");
				format_str.prepend("yyyy ");
			}
			TimeFormat *ptf = (TimeFormat*) alloc_tf_t::malloc();
			new (ptf) TimeFormat(format_str, prepend_year_to_value);
			TimeFormat_p tf(ptf, &PVDateTimeParser::destroy_tf);
			tf->current_year = _current_year;
			_time_format.push_back(tf);
		}
	}

	own_tfs = true;
}

void PVCore::PVDateTimeParser::copy(const PVDateTimeParser& src)
{
	//own_tfs = false;
	_time_format = src._time_format;
	_current_year = src._current_year;
	_last_match_time_format = src._last_match_time_format;
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
	local_parser(icuFromQStringAlias(time_format), _err)
{
	prepend_year_value = prepend_year;

	int32_t nlocales;
	const Locale* list_locales = Locale::getAvailableLocales(nlocales);
	parsers.reserve(nlocales);
	UnicodeString pattern = icuFromQStringAlias(time_format);

	for (int il = 0; il < nlocales; il++) {
		const Locale *cur_loc = &list_locales[il];
		UErrorCode err = U_ZERO_ERROR;
		SimpleDateFormat* sdf = new SimpleDateFormat(pattern, Locale::createFromName(cur_loc->getName()), err);
		if (U_SUCCESS(err)) {
			parsers.push_back(sdf);
		}
		else {
			PVLOG_WARN("Unable to create parser for locale %s.\n", cur_loc->getName());
		}
	}
	own_parsers = true;

	// The "local" parser is the one using the system's local
	//local_parser = SimpleDateFormat(pattern, err);
	last_good_parser = &local_parser;
}

PVCore::PVDateTimeParser::TimeFormat::TimeFormat(const TimeFormat& src) :
	local_parser(src.local_parser)
{
	parsers = src.parsers;
	current_year = src.current_year;
	last_good_parser = &local_parser;
	own_parsers = false;
}

PVCore::PVDateTimeParser::TimeFormat::~TimeFormat()
{
	if (!own_parsers)
		return;
	std::vector<SimpleDateFormat*>::iterator it;
	for (it = parsers.begin(); it != parsers.end(); it++) {
		delete *it;
	}
}

PVCore::PVDateTimeParser::TimeFormat& PVCore::PVDateTimeParser::TimeFormat::operator=(const TimeFormat& src)
{
	parsers = src.parsers;
	local_parser = src.local_parser;
	current_year = src.current_year;
	// Set the last_good_parser to the "local parser" of this object
	last_good_parser = &local_parser;
	own_parsers = false;

	return *this;
}

bool PVCore::PVDateTimeParser::TimeFormat::to_datetime(UnicodeString const& value, Calendar* cal)
{
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

	std::vector<SimpleDateFormat*>::iterator it;
	for (it = parsers.begin(); it != parsers.end(); it++) {
		SimpleDateFormat *cur_parser = *it;
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
	QString tmp = QString::fromRawData(reinterpret_cast<const QChar *>(value.getBuffer()), value.length());
	bool ok = false;
	UDate date = tmp.toLongLong(&ok);
	if (!ok)
		return false;
	UErrorCode err = U_ZERO_ERROR;
	cal->setTime(date, err);

	return U_SUCCESS(err);
}
