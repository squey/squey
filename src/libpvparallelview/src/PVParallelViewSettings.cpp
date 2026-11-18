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

#include <pvparallelview/PVParallelViewSettings.h>

#include <pvparallelview/PVScatterThumbnailsModel.h>

const PVCore::PVSetting<int> PVParallelView::settings::scatter_thumbnail_size(
    {.key = "scatter_thumbnails.size",
     .type = PVCore::PVSettingType::Int,
     .default_value = PVScatterThumbnailsModel::default_size,
     .title = "Thumbnail size",
     .description = "Width and height of the thumbnails as the gallery opens. The size slider "
                    "of the gallery changes it.",
     .minimum = PVScatterThumbnailsModel::min_size,
     .maximum = PVScatterThumbnailsModel::max_size,
     .unit = "px",
     .keywords = {"scatter", "thumbnail_size"}});

const PVCore::PVSetting<bool> PVParallelView::settings::scatter_thumbnails_by_correlation(
    {.key = "scatter_thumbnails.sort_by_correlation",
     .type = PVCore::PVSettingType::Bool,
     .default_value = true,
     .title = "Sort by correlation",
     .description = "Show the pairs of axes from the most correlated to the least, rather than "
                    "in the order of the axes. The sort list of the gallery changes it.",
     .keywords = {"scatter", "order"}});
