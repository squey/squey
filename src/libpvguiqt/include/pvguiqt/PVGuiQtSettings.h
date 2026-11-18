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

#ifndef PVGUIQT_PVGUIQTSETTINGS_H
#define PVGUIQT_PVGUIQTSETTINGS_H

#include <pvguiqt/export.h>
#include <pvkernel/core/PVSettings.h>

/**
 * The settings of libpvguiqt, and of the application around it. Their
 * descriptions are what the settings dialog shows.
 */
namespace PVGuiQt::settings
{

extern PVGUIQT_EXPORT const PVCore::PVSetting<bool> show_changelog;

/** In bytes, from PVAbstractTableModel::DISPLAYED_SIZE_MAX up. */
extern PVGUIQT_EXPORT const PVCore::PVSetting<int> listing_max_cell_size;
extern PVGUIQT_EXPORT const PVCore::PVSetting<bool> listing_links;

/** As the dialogs of distinct values and of aggregations open. */
extern PVGUIQT_EXPORT const PVCore::PVSetting<bool> statistics_follow_selection;
extern PVGUIQT_EXPORT const PVCore::PVSetting<bool> statistics_count;
extern PVGUIQT_EXPORT const PVCore::PVSetting<bool> statistics_percentage;
extern PVGUIQT_EXPORT const PVCore::PVSetting<bool> statistics_scientific;

extern PVGUIQT_EXPORT const PVCore::PVSetting<int> sql_result_rows;
extern PVGUIQT_EXPORT const PVCore::PVSetting<int> sql_history_size;

/** Of the code editors: the SQL console, the Python scripts, the JSON of the settings. */
extern PVGUIQT_EXPORT const PVCore::PVSetting<QString> editor_font_family;
/** In points. */
extern PVGUIQT_EXPORT const PVCore::PVSetting<double> editor_font_size;

/** From 10 to 100, a percentage of the available memory. */
extern PVGUIQT_EXPORT const PVCore::PVSetting<int> import_memory_warning_percent;

} // namespace PVGuiQt::settings

#endif // PVGUIQT_PVGUIQTSETTINGS_H
