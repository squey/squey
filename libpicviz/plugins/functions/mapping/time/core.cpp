/*
 * $Id$
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#include <pvkernel/core/general.h>

#include <QDate>
#include <QDateTime>
#include <QString>
#include <QStringList>

#include <QLocale>

#include "core.h"


#ifdef WIN32
inline struct tm* localtime_r (const time_t *clock, struct tm *result) { 
	if (!clock || !result) return NULL;
	memcpy(result,localtime(clock),sizeof(*result)); 
	return result; 
}
#endif

// Give me a time, give me a format, I send you a struct tm* back
// Used by our time functions
struct tm *picviz_mapping_time_to_tm(QStringList time_format, QString value, struct tm *tm_buf)
{
	time_t t;
	struct tm *tm;
	int count;

	QDateTime datetime;
	bool datetime_has_valid = 0;

	QDate current_date = QDate::currentDate();
	QString match_time_format_str;
	QLocale time_locale(QLocale::English, QLocale::UnitedStates);
	datetime.setTimeSpec(Qt::UTC);

	for (count=0; count < time_format.count(); count++) {
		QString format_str(time_format[count]);
		bool is_epoch = (format_str.compare("epoch") == 0);
		if (!is_epoch && !format_str.contains(QChar('y'))) {
			PVLOG_DEBUG("This string does not contain a date, adding one\n");
			QString year = current_date.toString("yyyy");
			year.append(" ");
			if (!value.startsWith(year)) {
				value.prepend(year);
			}
			if (!format_str.startsWith("yyyy ")) {
				format_str.prepend("yyyy ");
			}
		}
		//PVLOG_ERROR("TESTING value='%s'; format='%s'\n", value.toUtf8().data(), format_str.toUtf8().data());

		//datetime = QDateTime::fromString(value, format_str);
		if (is_epoch) {
			bool conv_ok;
			qint64 epoch_s = value.toLongLong(&conv_ok);
			if (conv_ok) {
				datetime.setTime_t(epoch_s); 
			}
			else {
				PVLOG_DEBUG("(%s): epoch_timeformat: unable to convert %s to an int64\n", __FUNCTION__, qPrintable(value));
			}
		}
		else {
			datetime = time_locale.toDateTime(value, format_str);
		}

		if (datetime.isValid()) {
			datetime_has_valid = 1;
			break;
		} else {
			match_time_format_str = format_str;
		}

	}

	if (!datetime_has_valid) {
		PVLOG_ERROR("(%s): invalid date or time (%s) with format (%s), returning 0!\n", __FUNCTION__, value.toUtf8().data(), match_time_format_str.toUtf8().data());
		//for (int i = 0; i < time_format.count(); i++) {
		//	PVLOG_ERROR("%s\n",qPrintable(time_format[i]));
		//}
		return NULL;
	}

	if (!datetime.isValid()) {
		PVLOG_ERROR("(%s): Discovered date or time invalid, this is a severe bug, we must not be there! returning 0!\n", __FUNCTION__);
		return NULL;
	}


	t = datetime.toTime_t();
	tm = localtime_r(&t, tm_buf);

	return tm;
}
