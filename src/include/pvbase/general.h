/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVBASE_GENERAL_H
#define PVBASE_GENERAL_H

#include <limits>

#include <QtGlobal>

// "SQUEY_VERSION_FILE_PATH" and "SQUEY_BUILD_FILE_PATH" are set by cmake
// (CMakeOptions.txt/CMakeVersionHandler.txt respectively)
#include SQUEY_VERSION_FILE_PATH
#include SQUEY_BUILD_FILE_PATH

#include "types.h"
#include "export.h"

static constexpr const char* SQUEY_ORGANISATION = "SQUEY";
static constexpr const char* SQUEY_APPLICATIONNAME = "Squey";

static constexpr const int PVEXTRACT_NUMBER_LINES_FIRST_DEFAULT = 1000000;
static constexpr const int PVEXTRACT_NUMBER_LINES_NEXT_DEFAULT = 1000000;

static constexpr const int FORMATBUILDER_EXTRACT_START_DEFAULT = 0;
static constexpr const int FORMATBUILDER_EXTRACT_END_DEFAULT = 100;

static constexpr const char* SQUEY_LOCAL_FORMAT_STR = "Default local format";
static constexpr const char* SQUEY_BROWSE_FORMAT_STR = "Custom format";

static constexpr const char PVCORE_DIRECTORY_SEP = ';';

static constexpr const char* PVCONFIG_FORMATS_INVALID_IGNORED = "formats/invalid/ignored";
static constexpr const char* PVCONFIG_FORMATS_SHOW_INVALID = "formats/invalid/warning";
#define PVCONFIG_FORMATS_SHOW_INVALID_DEFAULT (QVariant(true))

#define ALL_FILES_FILTER "All files (*.*)"

static constexpr const uint32_t SQUEY_ARCHIVES_VERSION = 3;

#define SQUEY_PATH_SEPARATOR "/"
static constexpr const char SQUEY_PATH_SEPARATOR_CHAR = '/';

static constexpr const char* SQUEY_DLL_EXTENSION = ".so";

static constexpr const char* SQUEY_DLL_PREFIX = "lib";

static constexpr const char* ESCAPE_PERCENT = "\%";

#define SQUEY_CONFDIR ".squey"
#define SQUEY_SQUEY_CONFDIR SQUEY_CONFDIR SQUEY_PATH_SEPARATOR "squey"

#endif /* PVBASE_GENERAL_H */
