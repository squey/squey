#include <pvkernel/core/PVDateTimeParser.h>
#include <QDate>

#include <boost/bind.hpp>

#include <tbb/tick_count.h>
#include <tbb/tbb_allocator.h>

/*boost::object_pool<PVCore::PVDateTimeParser::TimeFormat> PVCore::PVDateTimeParser::_alloc_tf;
boost::object_pool<PVCore::PVDateTimeParser::TimeFormatEpoch> PVCore::PVDateTimeParser::_alloc_tfe;
boost::object_pool<SimpleDateFormat> PVCore::PVDateTimeParser::TimeFormat::_alloc_df;*/

// No copy is made. The QString must remain valid as long as the UnicodeString object is !
UnicodeString PVCore::PVDateTimeParser::icuFromQStringAlias(const QString& src)
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
	//_alloc_tf.destroy(p);
}

void PVCore::PVDateTimeParser::destroy_tfe(TimeFormatEpoch* p)
{
	//_alloc_tfe.destroy(p);
}

PVCore::PVDateTimeParser::PVDateTimeParser(QStringList const& time_format):
	_org_time_format(time_format)
{
	// Compute the current year
	// If we are very unlucky, the year can change between this constructor and the computation of the struct tm's... !
	QString current_year = QDate::currentDate().toString("yyyy") + QString(" ");
	_current_year = UnicodeString((UChar*) current_year.data(), current_year.size());

	_last_match_time_format = NULL;
	_time_format.resize(time_format.size());
	static tbb::tbb_allocator<TimeFormatEpoch> alloc_epoch;
	static tbb::tbb_allocator<TimeFormat> alloc_format;
#pragma openmp parallel for
	for (int i = 0; i < time_format.size(); i++) {
		QString const& format_org = time_format.at(i);
		bool is_epoch = (format_org.compare("epoch") == 0);
		if (is_epoch) {
			//TimeFormatEpoch_p tf(_alloc_tfe.construct(), boost::bind(&PVDateTimeParser::destroy_tfe, _1));
			//TimeFormatEpoch* tf = _alloc_tfe.construct();
			TimeFormatEpoch* tf = alloc_epoch.allocate(1);
			new (tf) TimeFormatEpoch();
			_time_format[i] = tf;
		}
		else {
			bool prepend_year_to_value = !format_org.contains(QChar('y'));
			TimeFormat* tf = alloc_format.allocate(1);
			if (prepend_year_to_value) {
				QString format_str(format_org);
				PVLOG_DEBUG("This string does not contain a year, adding one\n");
				format_str.prepend("yyyy ");
				new (tf) TimeFormat(format_str, true);
			}
			else {
				//tf = _alloc_tf.construct(format_org, prepend_year_to_value);
				new (tf) TimeFormat(format_org, false);
			}
			tf->current_year = _current_year;
			_time_format[i] = tf;
		}
	}
}

PVCore::PVDateTimeParser::~PVDateTimeParser()
{
	static tbb::tbb_allocator<TimeFormatEpoch> alloc_epoch;
	static tbb::tbb_allocator<TimeFormat> alloc_format;
	list_time_format::const_iterator it;
	for (it = _time_format.begin(); it != _time_format.end(); it++) {
		TimeFormatInterface* tfi = *it;
		TimeFormat* tf = dynamic_cast<TimeFormat*>(tfi);
		if (tf) {
			tf->~TimeFormat();
			alloc_format.deallocate(tf, 1);
		}
		else {
			TimeFormatEpoch* tfe = (TimeFormatEpoch*) tfi;
			tfe->~TimeFormatEpoch();
			alloc_epoch.deallocate(tfe, 1);
		}
	}
}

void PVCore::PVDateTimeParser::copy(const PVDateTimeParser& src)
{
	list_time_format::const_iterator it;
	for (it = src._time_format.begin(); it != src._time_format.end(); it++) {
		// Use RTII to find out the real type of the TimeFormatInterface object.
		TimeFormatInterface* tfi = *it;
		TimeFormat* tf = dynamic_cast<TimeFormat*>(tfi);
		if (tf == NULL) {
			//TimeFormatEpoch_p ptfe(_alloc_tfe.construct(), boost::bind(&PVDateTimeParser::destroy_tfe, _1));
			TimeFormatEpoch_p ptfe = _alloc_tfe.construct();
			_time_format.push_back(ptfe);
		}
		else {
			//TimeFormat_p ptf(_alloc_tf.construct(*tf), boost::bind(&PVDateTimeParser::destroy_tf, _1));
			TimeFormat_p ptf = _alloc_tf.construct(*tf);
			_time_format.push_back(ptf);
		}
	}

	_current_year = src._current_year;
	_last_match_time_format = *(_time_format.begin());
	_org_time_format = src._org_time_format;
}

bool PVCore::PVDateTimeParser::mapping_time_to_cal(UnicodeString const& v, Calendar* cal)
{
	if (_last_match_time_format) {
		if (_last_match_time_format->to_datetime(v, cal))
			return true;
	}

	PVLOG_DEBUG("(PVDateTimeParser::mapping_time_to_cal) last known time format didn't match. Trying the other ones...\n");
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
	_parsers = NULL;
	_nparsers = 0;
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
	UnicodeString pattern = icuFromQStringAlias(time_format);

	static tbb::tbb_allocator<SimpleDateFormat> alloc;
	_parsers = alloc.allocate(nlocales);
	_nparsers = nlocales;
	for (int il = 0; il < nlocales; il++) {
		const Locale &cur_loc = list_locales[il];
		UErrorCode err = U_ZERO_ERROR;
		// MSVC seems not to be able to take the good constructor for SimpleDateFormat...
		SimpleDateFormat *psdf = &_parsers[il];
		new (psdf) SimpleDateFormat(pattern, Locale::createFromName(cur_loc.getName()), err);
		//SimpleDateFormat_p sdf(psdf, boost::bind(&TimeFormat::destroy_sdf, _1));
		if (!U_SUCCESS(err)) {
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
	static tbb::tbb_allocator<SimpleDateFormat> alloc;
	if (_parsers) {
		for (size_t i = 0; i < _nparsers; i++) {
			SimpleDateFormat* psdf = &_parsers[i];
			psdf->~SimpleDateFormat();
		}
		alloc.deallocate(_parsers, _nparsers);
	}
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
	//p->~SimpleDateFormat();
	//_alloc_df.free(p);
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
	last_good_parser->parse(value_, *cal, pos);

	if (pos.getErrorIndex() == -1) {
		return true;
	}

	PVLOG_DEBUG("(PVDateTimeParser::TimeFormat::to_datetime) last known parser (locale: %s) for current time format wasn't successful. Searching for a good one...\n", last_good_parser->getSmpFmtLocale().getName());

	for (size_t i = 0; i < _nparsers; i++) {
		SimpleDateFormat *cur_parser = &_parsers[i];
		if (cur_parser->getSmpFmtLocale() == last_good_parser->getSmpFmtLocale()) {
			continue;
		}
		ParsePosition pos_(0);
		cur_parser->parse(value_, *cal, pos_);
		if (pos_.getErrorIndex() == -1) {
			PVLOG_DEBUG("(PVDateTimeParser::TimeFormat::to_datetime) locale found: %s. This will be the next parser.\n", cur_parser->getSmpFmtLocale().getName());
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
	UDate date = (UDate) (tmp.toDouble(&ok) * 1000.0);
	if (!ok) {
		return false;
	}
	err = U_ZERO_ERROR;
	cal->setTime(date, err);

	return U_SUCCESS(err);
}
