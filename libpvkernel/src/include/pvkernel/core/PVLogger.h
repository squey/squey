/*
 * $Id: PVLogger.h 3090 2011-06-09 04:59:46Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 */

#ifndef PVCORE_PVLOGGER_H
#define PVCORE_PVLOGGER_H

#include <cstdio>

#include <QByteArray>
#include <QMutex>
#include <QString>

#include <pvbase/export.h>


#ifdef WIN32_DISABLE__
#define PVLOG_FATAL(fmt, ...) {}
#define PVLOG_ERROR(fmt, ...) {}
#define PVLOG_CUDA_ERROR(fmt, ...) {}
#define PVLOG_WARN(fmt, ...) {}
#define PVLOG_INFO(fmt, ...) {}
#define PVLOG_DEBUG(fmt, ...) {}
#define PVLOG_HEAVYDEBUG(fmt, ...) {}
#define PVLOG_PLAIN(fmt, ...) {}
#else
// The following are given be decreasing order of importance/verbosity
// Rem : Use HEAVYDEBUG level when you know that the message is potentially produced
//       a lot of times (as in a loop) and don't want it to appear in an everyday
//       DEBUG level.s
#define PVLOG_FATAL(fmt, ...) PVCore::PVLogger::getInstance()->fatal(fmt, ##__VA_ARGS__) 
#define PVLOG_ERROR(fmt, ...) PVCore::PVLogger::getInstance()->error(fmt, ##__VA_ARGS__) 
#define PVLOG_CUDA_ERROR(fmt, ...) PVCore::PVLogger::getInstance()->cudaError(fmt, ##__VA_ARGS__) 
#define PVLOG_WARN(fmt, ...) PVCore::PVLogger::getInstance()->warn(fmt, ##__VA_ARGS__) 
#define PVLOG_INFO(fmt, ...) PVCore::PVLogger::getInstance()->info(fmt, ##__VA_ARGS__) 
#define PVLOG_DEBUG(fmt, ...) PVCore::PVLogger::getInstance()->debug(fmt, ##__VA_ARGS__) 
#define PVLOG_HEAVYDEBUG(fmt, ...) PVCore::PVLogger::getInstance()->heavydebug(fmt, ##__VA_ARGS__)
// The next MACRO outputs the given message without prefixing it with the usual stuff (works as a printf)
#define PVLOG_PLAIN(fmt, ...) PVCore::PVLogger::getInstance()->plain(fmt, ##__VA_ARGS__) 
#endif

namespace PVCore {

	class LibCoreDecl PVLogger {
	private:
		FILE *fp;
		QByteArray log_filename;
		QMutex mutex;
	protected:
		PVLogger();
	public:
		enum LogLevel {
			PVLOG_FATAL,
			PVLOG_ERROR,
			PVLOG_CUDA_ERROR,
			PVLOG_WARN,
			PVLOG_INFO,
			PVLOG_DEBUG,
			PVLOG_HEAVYDEBUG,
		};

		~PVLogger();

		/* Singleton */
		static PVLogger *getInstance();

		LogLevel level;
		QString datetime_format;

		void heavydebug(const char *, ...);	
		void debug(const char *, ...);
		void info(const char *, ...);
		void warn(const char *, ...);
		void error(const char *, ...);
		void cudaError(const char *, ...);
		void fatal(const char *, ...);
		void plain(const char *, ...);

		QString get_now_str();
	};
}

#endif	/* PVCORE_PVLOGGER_H */
