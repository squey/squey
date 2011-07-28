/*
 * $Id: general.h 3221 2011-06-30 11:45:19Z aguinet $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#ifndef PVCORE_GENERAL_H
#define PVCORE_GENERAL_H

#include <QtGlobal>
#include <QSettings>

#include "types.h"
#include "export.h"
#include "PVLogger.h"

extern PVCore::PVLogger pvlog;
static QSettings pvconfig(QString("pvconfig.ini"), QSettings::IniFormat);

#define PICVIZ_VERSION_STR "2.1.0"
/*
 * PVCORE_VERSION is (major << 16) + (minor << 8) + patch.
 */
/* #define PVCORE_VERSION 0x010102 for 1.1.2 */
#define PVCORE_VERSION 0x020100

/*
 * Use it like this: if (PVCORE_VERSION >= PVCORE_VERSION_CHECK(1, 1, 2))
 */
#define PVCORE_VERSION_CHECK(major, minor, patch) ((major<<16)|(minor<<8)|(patch))


#define PVCORE_QVARIANT_METATYPE_HEIGHT 30
#define PVCORE_QVARIANT_METATYPE_WIDTH 350


#define picviz_max(x,y) ((x)>(y)?(x):(y))
#define picviz_min(x,y) ((x)<(y)?(x):(y))

#define PICVIZ_LINES_MAX 5000000

#define PICVIZ_EVENTLINE_LINES_MAX PICVIZ_LINES_MAX
#define PICVIZ_AXES_MAX 8096 // Max number of axes
#define PICVIZ_FIELD_MAX 8096 // Max value a string can receive as a field

#define PVEXTRACT_NUMBER_LINES_FIRST_DEFAULT 1000000
#define PVEXTRACT_NUMBER_LINES_NEXT_DEFAULT 1000000

#define FORMATBUILDER_EXTRACT_START_DEFAULT 0
#define FORMATBUILDER_EXTRACT_END_DEFAULT 100

#define PVFORMAT_NUMBER_FIELD_URL 6
#define PVFORMAT_NUMBER_FIELD_PCAP 8

#define PICVIZ_AUTOMATIC_FORMAT_STR "automatic"

#define PVCORE_DIRECTORY_SEP ';'

#ifdef WIN32
#define PICVIZ_PATH_SEPARATOR "\\"
#define PICVIZ_PATH_SEPARATOR_CHAR '\\'
#else
#define PICVIZ_PATH_SEPARATOR "/"
#define PICVIZ_PATH_SEPARATOR_CHAR '/'
#endif

#ifdef WIN32
#define PICVIZ_DLL_EXTENSION ".dll"
#else
#define PICVIZ_DLL_EXTENSION ".so"
#endif

#ifdef WIN32
#define PICVIZ_DLL_PREFIX ""
#else
#define PICVIZ_DLL_PREFIX "lib"
#endif

#ifdef WIN32
#define ESCAPE_PERCENT "%%"
#else
#define ESCAPE_PERCENT "\%"
#endif

#endif	/* PVCORE_GENERAL_H */
