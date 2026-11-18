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

#ifndef PVPARALLELVIEW_PVPARALLELVIEWSETTINGS_H
#define PVPARALLELVIEW_PVPARALLELVIEWSETTINGS_H

#include <pvkernel/core/PVSettings.h>
#include <pvparallelview/export.h>

/**
 * The settings of libpvparallelview. Their descriptions are what the settings
 * dialog shows.
 */
namespace PVParallelView::settings
{

/** In pixels, within PVScatterThumbnailsModel::min_size and max_size. */
extern PVPARALLELVIEW_EXPORT const PVCore::PVSetting<int> scatter_thumbnail_size;
extern PVPARALLELVIEW_EXPORT const PVCore::PVSetting<bool> scatter_thumbnails_by_correlation;

} // namespace PVParallelView::settings

#endif // PVPARALLELVIEW_PVPARALLELVIEWSETTINGS_H
