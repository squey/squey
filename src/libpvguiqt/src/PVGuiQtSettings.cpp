//
// MIT License
//
// © Squey, 2026
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvguiqt/PVGuiQtSettings.h>

#include <pvguiqt/PVAbstractTableModel.h>

const PVCore::PVSetting<bool> PVGuiQt::settings::show_changelog(
    {.key = "general.show_changelog",
     .type = PVCore::PVSettingType::Bool,
     .default_value = true,
     .title = "Changes of a new version",
     .description = "Show what changed, the first time a new version of Squey starts.",
     .keywords = {"changelog", "update", "release notes"}});

const PVCore::PVSetting<int> PVGuiQt::settings::listing_max_cell_size(
    {.key = "listing.max_cell_size",
     .type = PVCore::PVSettingType::Int,
     .default_value = int(PVAbstractTableModel::DISPLAYED_SIZE_MAX),
     .title = "Longest value shown",
     .description = "Bytes of a value that a cell and its tooltip show, the rest being cut. "
                    "Longer values make the tables slower to draw.",
     // Below, a web address could be cut, and the link open another one.
     .minimum = int(PVAbstractTableModel::DISPLAYED_SIZE_MAX),
     .maximum = 65536,
     .unit = "bytes",
     .keywords = {"cell", "tooltip", "length"}});

const PVCore::PVSetting<bool> PVGuiQt::settings::listing_links(
    {.key = "listing.links",
     .type = PVCore::PVSettingType::Bool,
     .default_value = true,
     .title = "Web addresses as links",
     .description = "Show the values that are web addresses as links, which a click opens in "
                    "the browser.",
     .keywords = {"URL", "hyperlink", "http"}});

const PVCore::PVSetting<int> PVGuiQt::settings::import_memory_warning_percent(
    {.key = "import.memory_warning_percent",
     .type = PVCore::PVSettingType::Int,
     .default_value = 80,
     .title = "Memory warning",
     .description = "Share of the available memory that an import may be expected to take "
                    "before Squey pauses it, and asks whether to go on.",
     .minimum = 10,
     .maximum = 100,
     .unit = "%",
     .keywords = {"RAM"}});
