//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/PVLogger.h> // for PVLogger, etc
#include <qtenvironmentvariables.h>
#include <QDateTime>
#include <QString>
#include <QByteArray>
#include <cstdarg> // for va_end, va_list, va_start
#include <cstdio>  // for fprintf, fclose, fopen
#include <iostream>

PVCore::PVLogger::PVLogger()
{
	QByteArray log_level;

	log_filename = qgetenv("SQUEY_LOG_FILE");
	datetime_format = "yyyy-MM-dd hh:mm:ss.zzz";

	if (!log_filename.isEmpty()) {
		fp = fopen(log_filename.constData(), "a");
	}

	level = PVLOG_INFO;
	log_level = qgetenv("SQUEY_DEBUG_LEVEL");
	if (!log_level.isEmpty()) {
		if (log_level == QString("FATAL")) {
			level = PVLOG_FATAL;
		}
		if (log_level == QString("ERROR")) {
			level = PVLOG_ERROR;
		}
		if (log_level == QString("WARN")) {
			level = PVLOG_WARN;
		}
		if (log_level == QString("INFO")) {
			level = PVLOG_INFO;
		}
		if (log_level == QString("DEBUG")) {
			level = PVLOG_DEBUG;
		}
		if (log_level == QString("HEAVYDEBUG")) {
			level = PVLOG_HEAVYDEBUG;
		}
	}
}

PVCore::PVLogger::~PVLogger()
{
	if (!log_filename.isEmpty()) {
		fclose(fp);
	}
}

QString PVCore::PVLogger::get_now_str()
{
	QDateTime now = QDateTime::currentDateTime();
	return now.toString(datetime_format);
}

void PVCore::PVLogger::heavydebug(const char* format, ...)
{
	QString res;
	va_list ap;

	if (level < PVLOG_HEAVYDEBUG)
		return;

	va_start(ap, format);
	res = QString::vasprintf(format, ap);

	if (log_filename.isEmpty()) {
		std::cerr << qPrintable(get_now_str()) << " *** HEAVYDEBUG *** " << qPrintable(res);
	} else {
		fprintf(fp, "%s *** HEAVYDEBUG *** %s", qPrintable(get_now_str()), qPrintable(res));
	}

	va_end(ap);
}

void PVCore::PVLogger::debug(const char* format, ...)
{
	QString res;
	va_list ap;

	if (level < PVLOG_DEBUG)
		return;

	va_start(ap, format);
	res = QString::vasprintf(format, ap);

	if (log_filename.isEmpty()) {
		std::cerr << qPrintable(get_now_str()) << " *** DEBUG *** " << qPrintable(res);
	} else {
		fprintf(fp, "%s *** DEBUG *** %s", qPrintable(get_now_str()), qPrintable(res));
	}

	va_end(ap);
}

void PVCore::PVLogger::info(const char* format, ...)
{
	QString res;
	va_list ap;

	if (level < PVLOG_INFO)
		return;

	mutex.lock();

	va_start(ap, format);
	res = QString::vasprintf(format, ap);

	if (log_filename.isEmpty()) {
		std::cerr << qPrintable(get_now_str()) << " *** INFO *** " << qPrintable(res);
	} else {
		fprintf(fp, "%s *** INFO *** %s", qPrintable(get_now_str()), qPrintable(res));
	}

	va_end(ap);

	mutex.unlock();
}

void PVCore::PVLogger::warn(const char* format, ...)
{
	QString res;
	va_list ap;

	if (level < PVLOG_WARN)
		return;

	va_start(ap, format);
	res = QString::vasprintf(format, ap);

	if (log_filename.isEmpty()) {
		std::cerr << qPrintable(get_now_str()) << " *** WARN *** " << qPrintable(res);
	} else {
		fprintf(fp, "%s *** WARN *** %s", qPrintable(get_now_str()), qPrintable(res));
	}

	va_end(ap);
}

void PVCore::PVLogger::error(const char* format, ...)
{
	QString res;
	va_list ap;

	if (level < PVLOG_ERROR)
		return;

	va_start(ap, format);
	res = QString::vasprintf(format, ap);

	if (log_filename.isEmpty()) {
		std::cerr << qPrintable(get_now_str()) << " *** ERROR *** " << qPrintable(res);
	} else {
		fprintf(fp, "%s *** ERROR *** %s", qPrintable(get_now_str()), qPrintable(res));
	}

	va_end(ap);
}

void PVCore::PVLogger::fatal(const char* format, ...)
{
	QString res;
	va_list ap;

	if (level < PVLOG_FATAL)
		return;

	va_start(ap, format);
	res = QString::vasprintf(format, ap);

	if (log_filename.isEmpty()) {
		std::cerr << qPrintable(get_now_str()) << " *** FATAL *** " << qPrintable(res);
	} else {
		fprintf(fp, "%s *** FATAL *** %s", qPrintable(get_now_str()), qPrintable(res));
	}

	va_end(ap);
}

void PVCore::PVLogger::plain(const char* format, ...)
{
	QString res;
	va_list ap;

	if (level < PVLOG_INFO)
		return;

	va_start(ap, format);
	res = QString::vasprintf(format, ap);

	if (log_filename.isEmpty()) {
		std::cerr << qPrintable(res);
	} else {
		fprintf(fp, "%s", qPrintable(res));
	}

	va_end(ap);
}

PVCore::PVLogger* PVCore::PVLogger::getInstance()
{
	static PVCore::PVLogger instance;
	return &instance;
}
