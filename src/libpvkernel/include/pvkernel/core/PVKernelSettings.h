/* * MIT License
 *
 * © Squey, 2026
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

#ifndef PVCORE_PVKERNELSETTINGS_H
#define PVCORE_PVKERNELSETTINGS_H

#include <pvkernel/core/PVSettings.h>
#include <pvkernel/export.h>

#include <QString>

/**
 * The settings of libpvkernel. Their descriptions are what the settings dialog
 * shows.
 */
namespace PVCore::settings
{

/** "dark", "light", or "system" to follow the desktop. */
extern PVKERNEL_EXPORT const PVSetting<QString> color_scheme;

/**
 * Where the imported data and the temporary files go. Read once, as everything
 * written there is found again under it: see PVRush::PVNrawCacheManager.
 */
extern PVKERNEL_EXPORT const PVSetting<QString> working_directory;

/** Per list but the investigations, which keep them all. */
extern PVKERNEL_EXPORT const PVSetting<int> recent_items;

/** In bytes. */
extern PVKERNEL_EXPORT const PVSetting<int> import_chunk_size;

/** 0 for as many chunks as the processor runs threads. */
extern PVKERNEL_EXPORT const PVSetting<int> import_parallel_chunks;

/** What the CSV export last used, which the next one starts with. */
extern PVKERNEL_EXPORT const PVSetting<bool> export_csv_header;
/** One character each, the tabulation included. */
extern PVKERNEL_EXPORT const PVSetting<QString> export_csv_separator;
extern PVKERNEL_EXPORT const PVSetting<QString> export_csv_quote;

/**
 * "auto", "cpu" for the first OpenCL platform that draws with the processor, "qt", or
 * "gpu:" or "cpu:" before the name of an OpenCL platform, for its graphics cards or its
 * processors: see PVOpenCL::lines_platform().
 */
extern PVKERNEL_EXPORT const PVSetting<QString> lines_platform;

} // namespace PVCore::settings

#endif // PVCORE_PVKERNELSETTINGS_H
