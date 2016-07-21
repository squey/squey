/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVKERNEL_CORE_PVLOGGER_H
#define PVKERNEL_CORE_PVLOGGER_H

#include <cstdio>

#include <QByteArray>
#include <QMutex>
#include <QString>
#include <tuple>

#include <pvbase/export.h>

namespace PVCore
{

class PVLogger
{
  public:
	enum LogLevel {
		PVLOG_FATAL,
		PVLOG_ERROR,
		PVLOG_WARN,
		PVLOG_INFO,
		PVLOG_DEBUG,
		PVLOG_HEAVYDEBUG,
	};

  protected:
	PVLogger();

  public:
	~PVLogger();

	/* Singleton */
	static PVLogger* getInstance();

	LogLevel level;
	QString datetime_format;

	void heavydebug(const char*, ...);
	void debug(const char*, ...);
	void info(const char*, ...);
	void warn(const char*, ...);
	void error(const char*, ...);
	void fatal(const char*, ...);
	void plain(const char*, ...);

	QString get_now_str();

  private:
	FILE* fp;
	QByteArray log_filename;
	QMutex mutex;
};
} // namespace PVCore

// The following are given be decreasing order of importance/verbosity
// Rem : Use HEAVYDEBUG level when you know that the message is potentially produced
//       a lot of times (as in a loop) and don't want it to appear in an everyday
//       DEBUG level.
template <class T, class... U>
void PVLOG_FATAL(T&& fmt, U&&... u)
{
	PVCore::PVLogger::getInstance()->fatal(std::forward<T>(fmt), std::forward<U>(u)...);
}
template <class T, class... U>
void PVLOG_ERROR(T&& fmt, U&&... u)
{
	PVCore::PVLogger::getInstance()->error(std::forward<T>(fmt), std::forward<U>(u)...);
}
template <class T, class... U>
void PVLOG_WARN(T&& fmt, U&&... u)
{
	PVCore::PVLogger::getInstance()->warn(std::forward<T>(fmt), std::forward<U>(u)...);
}
template <class T, class... U>
void PVLOG_INFO(T&& fmt, U&&... u)
{
	PVCore::PVLogger::getInstance()->info(std::forward<T>(fmt), std::forward<U>(u)...);
}

// If we are in release mode, PVLOG_DEBUG and PVLOG_HEAVYDEBUG must not produce any code !
template <class T, class... U>
void PVLOG_DEBUG(T&& fmt, U&&... u)
{
#ifdef NDEBUG
	PVCore::PVLogger::getInstance()->debug(std::forward<T>(fmt), std::forward<U>(u)...);
#else
	(void)fmt;
	std::tuple<U...> __attribute((unused)) d{u...};
#endif
}
template <class T, class... U>
void PVLOG_HEAVYDEBUG(T&& fmt, U&&... u)
{
#ifdef NDEBUG
	PVCore::PVLogger::getInstance()->heavydebug(std::forward<T>(fmt), std::forward<U>(u)...);
#else
	(void)fmt;
	std::tuple<U...> __attribute((unused)) d{u...};
#endif
}
// The next MACRO outputs the given message without prefixing it with the usual stuff (works as a
// printf)
template <class T, class... U>
void PVLOG_PLAIN(T&& fmt, U&&... u)
{
	PVCore::PVLogger::getInstance()->plain(std::forward<T>(fmt), std::forward<U>(u)...);
}

#endif // PVKERNEL_CORE_PVLOGGER_H
