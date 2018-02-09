/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVBASE_GENERAL_H
#define PVBASE_GENERAL_H

#include <limits>

#include <QtGlobal>

// "INENDI_VERSION_FILE_PATH" and "INENDI_BUILD_FILE_PATH" are set by cmake
// (CMakeOptions.txt/CMakeVersionHandler.txt respectively)
#include INENDI_VERSION_FILE_PATH
#include INENDI_BUILD_FILE_PATH

#include "types.h"
#include "export.h"

static constexpr const char* INENDI_ORGANISATION = "ESI Group";
static constexpr const char* INENDI_APPLICATIONNAME = "INENDI Inspector";

static constexpr const char* INENDI_LICENSE_PATH = "~/.inendi/licenses/inspector.lic";
static constexpr const char* INENDI_LICENSE_PREFIX = "II";
static constexpr const char* INENDI_LICENSE_FEATURE = "INSPECTOR";
static constexpr const char* INENDI_LICENSE_MAXMEM = "MAXMEM";

static constexpr const int PVEXTRACT_NUMBER_LINES_FIRST_DEFAULT = 1000000;
static constexpr const int PVEXTRACT_NUMBER_LINES_NEXT_DEFAULT = 1000000;

static constexpr const int FORMATBUILDER_EXTRACT_START_DEFAULT = 0;
static constexpr const int FORMATBUILDER_EXTRACT_END_DEFAULT = 100;

static constexpr const char* INENDI_AUTOMATIC_FORMAT_STR = "[auto detection...]";
static constexpr const char* INENDI_LOCAL_FORMAT_STR = "[default local format]";
static constexpr const char* INENDI_BROWSE_FORMAT_STR = "[choose my format...]";

static constexpr const char PVCORE_DIRECTORY_SEP = ';';

static constexpr const char* PVCONFIG_FORMATS_INVALID_IGNORED = "formats/invalid/ignored";
static constexpr const char* PVCONFIG_FORMATS_SHOW_INVALID = "formats/invalid/warning";
#define PVCONFIG_FORMATS_SHOW_INVALID_DEFAULT (QVariant(true))

#define ALL_FILES_FILTER "All files (*.*)"

static constexpr const uint32_t INENDI_ARCHIVES_VERSION = 3;

#define INENDI_PATH_SEPARATOR "/"
static constexpr const char INENDI_PATH_SEPARATOR_CHAR = '/';

static constexpr const char* INENDI_DLL_EXTENSION = ".so";

static constexpr const char* INENDI_DLL_PREFIX = "lib";

static constexpr const char* ESCAPE_PERCENT = "\%";

#define INENDI_CONFDIR ".inendi"
#define INENDI_INSPECTOR_CONFDIR INENDI_CONFDIR INENDI_PATH_SEPARATOR "inspector"

#endif /* PVBASE_GENERAL_H */
