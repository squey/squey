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

#ifndef PVRUSH_PVFORMAT_TYPES_H
#define PVRUSH_PVFORMAT_TYPES_H

#include <pvbase/export.h>
#include <pvkernel/rush/PVAxisFormat.h>

#include <memory>

namespace PVRush
{

class PVFormat;
typedef std::shared_ptr<PVFormat> PVFormat_p;
typedef QMap<QString, PVRush::PVFormat> hash_formats;

// Axes properties' name
#define PVFORMAT_AXIS_NAME_STR "name"
#define PVFORMAT_AXIS_TYPE_STR "type"
#define PVFORMAT_AXIS_MAPPING_STR "mapping"
#define PVFORMAT_AXIS_SCALING_STR "scaling"
#define PVFORMAT_AXIS_COLOR_STR "color"
#define PVFORMAT_AXIS_TITLECOLOR_STR "titlecolor"
#define PVFORMAT_AXIS_TYPE_FORMAT_STR "type_format"

// Axes properties' default value
#define PVFORMAT_AXIS_NAME_DEFAULT ""
#define PVFORMAT_AXIS_TYPE_DEFAULT "string"
#define PVFORMAT_AXIS_MAPPING_DEFAULT "default"
#define PVFORMAT_AXIS_SCALING_DEFAULT "default"
#define PVFORMAT_AXIS_KEY_DEFAULT "false"
#define PVFORMAT_AXIS_COLOR_DEFAULT "#1a72bb"
#define PVFORMAT_AXIS_TITLECOLOR_DEFAULT PVFORMAT_AXIS_COLOR_DEFAULT
#define PVFORMAT_AXIS_TYPE_FORMAT_DEFAULT ""
#define PVFORMAT_AXIS_TIMESAMPLE_DEFAULT ""
#define PVFORMAT_AXES_COMBINATION_DEFAULT ""

// Filters properties' name
#define PVFORMAT_FILTER_TYPE_STR "type"
#define PVFORMAT_FILTER_NAME_STR "name"

// Filters properties' default value
#define PVFORMAT_FILTER_TYPE_DEFAULT ""
#define PVFORMAT_FILTER_NAME_DEFAULT ""

// Mapping/scaling properties'name
#define PVFORMAT_MAP_PLOT_MODE_STR "mode"
#define PVFORMAT_MAP_PLOT_MODE_DEFAULT "default"

// XML format tags
#define PVFORMAT_XML_TAG_FIELD_STR "field"
#define PVFORMAT_XML_TAG_AXIS_STR "axis"
#define PVFORMAT_XML_TAG_SPLITTER_STR "splitter"
#define PVFORMAT_XML_TAG_FILTER_STR "filter"
#define PVFORMAT_XML_TAG_CONVERTER_STR "converter"
#define PVFORMAT_XML_TAG_AXES_COMBINATION_STR "axes-combination"
#define PVFORMAT_XML_TAG_MAPPING "mapping"
#define PVFORMAT_XML_TAG_PLOTTING "plotting"
#define PVFORMAT_XML_TAG_SCALING "scaling"

// Format version
#define PVFORMAT_CURRENT_VERSION "9"

#define PVFORMAT_NUMBER_FIELD_URL 6
} // namespace PVRush

#endif
