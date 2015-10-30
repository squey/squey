/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVBASE_GENERAL_H
#define PVBASE_GENERAL_H

#include <QtGlobal>

// "PICVIZ_CFG_FILE_PATH", "PICVIZ_VERSION_FILE_PATH" and "PICVIZ_BUILD_FILE_PATH" are set by cmake (CMakeOptions.txt/CMakeVersionHandler.txt respectively)
#include PICVIZ_CFG_FILE_PATH
#include PICVIZ_VERSION_FILE_PATH
#include PICVIZ_BUILD_FILE_PATH

#include "types.h"
#include "export.h"

#define PICVIZ_ORGANISATION "Picviz Labs"
#define PICVIZ_APPLICATIONNAME "Picviz Inspector"

#define PVCORE_QVARIANT_METATYPE_HEIGHT 30
#define PVCORE_QVARIANT_METATYPE_WIDTH 350


#define picviz_max(x,y) ((x)>(y)?(x):(y))
#define picviz_min(x,y) ((x)<(y)?(x):(y))

#define PICVIZ_LINES_MAX CUSTOMER_LINESNUMBER

#define PICVIZ_EVENTLINE_LINES_MAX PICVIZ_LINES_MAX
#define PICVIZ_AXES_MAX 8096 // Max number of axes
#define PICVIZ_FIELD_MAX 8096 // Max value a string can receive as a field

#define PVEXTRACT_NUMBER_LINES_FIRST_DEFAULT 1000000
#define PVEXTRACT_NUMBER_LINES_NEXT_DEFAULT 1000000

#define FORMATBUILDER_EXTRACT_START_DEFAULT 0
#define FORMATBUILDER_EXTRACT_END_DEFAULT 100

#define PICVIZ_AUTOMATIC_FORMAT_STR "[auto detection...]"
#define PICVIZ_LOCAL_FORMAT_STR "[default local format]"
#define PICVIZ_BROWSE_FORMAT_STR "[choose my format...]"

#define PVCORE_DIRECTORY_SEP ';'

#define PVCONFIG_FORMATS_INVALID_IGNORED "formats/invalid/ignored"
#define PVCONFIG_FORMATS_SHOW_INVALID "formats/invalid/warning"
#define PVCONFIG_FORMATS_SHOW_INVALID_DEFAULT (QVariant(true))

#define ALL_FILES_FILTER "All files (*.*)"

#define PICVIZ_ARCHIVES_VERSION 2

#define PICVIZ_PATH_SEPARATOR "/"
#define PICVIZ_PATH_SEPARATOR_CHAR '/'

#define PICVIZ_DLL_EXTENSION ".so"

#define PICVIZ_DLL_PREFIX "lib"

#define ESCAPE_PERCENT "\%"

#define PICVIZ_CONFDIR ".picviz"
#define PICVIZ_INSPECTOR_CONFDIR PICVIZ_CONFDIR PICVIZ_PATH_SEPARATOR "inspector"

#define picviz_verify(e) __picviz_verify(e, __FILE__, __LINE__)
#define __picviz_verify(e, F, L)\
	if (!(e)) {\
		fprintf(stderr, "valid assertion failed at %s:%d: %s.\n", F, L, #e);\
		abort();\
	}

#define NEXT_MULTIPLE(n, align) ((((n)*(align)-1)/(align))*(align))
#define PREV_MULTIPLE(n, align) (((n)/(align))*(align))

#define PV_UNUSED(v) ((void)v)

#endif	/* PVBASE_GENERAL_H */
