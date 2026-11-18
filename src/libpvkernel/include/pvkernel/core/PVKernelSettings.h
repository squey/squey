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

/** In bytes. */
extern PVKERNEL_EXPORT const PVSetting<int> import_chunk_size;

/** 0 for as many chunks as the processor runs threads. */
extern PVKERNEL_EXPORT const PVSetting<int> import_parallel_chunks;

extern PVKERNEL_EXPORT const PVSetting<bool> force_cpu;

/** -1 for the first platform that can draw. */
extern PVKERNEL_EXPORT const PVSetting<int> opencl_platform;

} // namespace PVCore::settings

#endif // PVCORE_PVKERNELSETTINGS_H
