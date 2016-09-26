/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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
typedef QHash<QString, PVRush::PVFormat> hash_formats;

// Axes properties' name
#define PVFORMAT_AXIS_NAME_STR "name"
#define PVFORMAT_AXIS_TYPE_STR "type"
#define PVFORMAT_AXIS_MAPPING_STR "mapping"
#define PVFORMAT_AXIS_PLOTTING_STR "plotting"
#define PVFORMAT_AXIS_COLOR_STR "color"
#define PVFORMAT_AXIS_TITLECOLOR_STR "titlecolor"
#define PVFORMAT_AXIS_TYPE_FORMAT_STR "type_format"

// Axes properties' default value
#define PVFORMAT_AXIS_NAME_DEFAULT ""
#define PVFORMAT_AXIS_TYPE_DEFAULT "string"
#define PVFORMAT_AXIS_MAPPING_DEFAULT "default"
#define PVFORMAT_AXIS_PLOTTING_DEFAULT "default"
#define PVFORMAT_AXIS_KEY_DEFAULT "false"
#define PVFORMAT_AXIS_COLOR_DEFAULT "#ffffff"
#define PVFORMAT_AXIS_TITLECOLOR_DEFAULT "#ff921d"
#define PVFORMAT_AXIS_TYPE_FORMAT_DEFAULT ""
#define PVFORMAT_AXIS_TIMESAMPLE_DEFAULT ""
#define PVFORMAT_AXES_COMBINATION_DEFAULT ""

// Filters properties' name
#define PVFORMAT_FILTER_TYPE_STR "type"
#define PVFORMAT_FILTER_NAME_STR "name"

// Filters properties' default value
#define PVFORMAT_FILTER_TYPE_DEFAULT ""
#define PVFORMAT_FILTER_NAME_DEFAULT ""

// Mapping/plotting properties'name
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

// Format version
#define PVFORMAT_CURRENT_VERSION "7"

#define PVFORMAT_NUMBER_FIELD_URL 6
} // namespace PVRush

#endif
