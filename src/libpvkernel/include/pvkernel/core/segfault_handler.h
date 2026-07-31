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

#ifndef PVCORE_PRINTBACKTRACE_H
#define PVCORE_PRINTBACKTRACE_H

#include <pvkernel/export.h>

#include <string>

/**
 * Start the out-of-process Crashpad handler. Uploads stay disabled: the handler
 * spawns squey-crashreport, which asks for consent before sending anything.
 */
PVKERNEL_EXPORT void init_segfault_handler();

/**
 * Path of the crash report database, shared by squey and squey-crashreport.
 */
PVKERNEL_EXPORT std::string crash_report_database_path();

/**
 * Path of the most recent minidump waiting to be sent, empty if there is none.
 */
PVKERNEL_EXPORT std::string pending_crash_report_path();

/**
 * Drop a report from the database, once it has been sent or declined.
 */
PVKERNEL_EXPORT void discard_crash_report(const std::string& minidump_path);

#endif
