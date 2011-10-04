/*
 * $Id$
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#ifndef PVBASE_GENERAL_H
#define PVBASE_GENERAL_H

#include <QtGlobal>
#include <QSettings>

#include "types.h"
#include "export.h"
#include "version.h"

static QSettings pvconfig(QString("pvconfig.ini"), QSettings::IniFormat);

#define PICVIZ_ORGANISATION "Picviz Labs"
#define PICVIZ_APPLICATIONNAME "Picviz Inspector"


#define PVCORE_QVARIANT_METATYPE_HEIGHT 30
#define PVCORE_QVARIANT_METATYPE_WIDTH 350


#define picviz_max(x,y) ((x)>(y)?(x):(y))
#define picviz_min(x,y) ((x)<(y)?(x):(y))

#define PICVIZ_LINES_MAX 1000500

#define PICVIZ_EVENTLINE_LINES_MAX PICVIZ_LINES_MAX
#define PICVIZ_AXES_MAX 8096 // Max number of axes
#define PICVIZ_FIELD_MAX 8096 // Max value a string can receive as a field

#define PVEXTRACT_NUMBER_LINES_FIRST_DEFAULT 1000000
#define PVEXTRACT_NUMBER_LINES_NEXT_DEFAULT 1000000

#define FORMATBUILDER_EXTRACT_START_DEFAULT 0
#define FORMATBUILDER_EXTRACT_END_DEFAULT 100

#define PICVIZ_AUTOMATIC_FORMAT_STR "automatic"

#define PVCORE_DIRECTORY_SEP ';'

#define PVCONFIG_FORMATS_INVALID_IGNORED "formats/invalid/ignored"
#define PVCONFIG_FORMATS_SHOW_INVALID "formats/invalid/warning"
#define PVCONFIG_FORMATS_SHOW_INVALID_DEFAULT (QVariant(true))

#define PICVIZ_ARCHIVES_VERSION 1

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

#endif	/* PVBASE_GENERAL_H */
