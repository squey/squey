#ifndef PVRUSH_PVFORMAT_TYPES_H
#define PVRUSH_PVFORMAT_TYPES_H

#include <pvbase/export.h>
#include <boost/shared_ptr.hpp>
#include <pvkernel/rush/PVAxisFormat.h>

namespace PVRush {

class PVFormat;
typedef boost::shared_ptr<PVFormat> PVFormat_p;
typedef QList<PVAxisFormat> list_axes_t;
typedef QHash<QString, PVRush::PVFormat> hash_formats;

// Axes properties' name
#define PVFORMAT_AXIS_NAME_STR "name"
#define PVFORMAT_AXIS_TYPE_STR "type"
#define PVFORMAT_AXIS_MAPPING_STR "mapping"
#define PVFORMAT_AXIS_PLOTTING_STR "plotting"
#define PVFORMAT_AXIS_KEY_STR "key"
#define PVFORMAT_AXIS_GROUP_STR "group"
#define PVFORMAT_AXIS_COLOR_STR "color"
#define PVFORMAT_AXIS_TITLECOLOR_STR "titlecolor"
#define PVFORMAT_AXIS_TIMEFORMAT_STR "time-format"
#define PVFORMAT_AXIS_TIMESAMPLE_STR "time-sample"
// TODO: change by "tag" when that will be done
#define PVFORMAT_AXIS_TAG_STR "name"

// Axes properties' default value
#define PVFORMAT_AXIS_NAME_DEFAULT ""
#define PVFORMAT_AXIS_TYPE_DEFAULT "enum"
#define PVFORMAT_AXIS_MAPPING_DEFAULT "default"
#define PVFORMAT_AXIS_PLOTTING_DEFAULT "default"
#define PVFORMAT_AXIS_KEY_DEFAULT "false"
#define PVFORMAT_AXIS_GROUP_DEFAULT ""
#define PVFORMAT_AXIS_COLOR_DEFAULT "#ffffff"
#define PVFORMAT_AXIS_TITLECOLOR_DEFAULT "#ff921d"
#define PVFORMAT_AXIS_TIMEFORMAT_DEFAULT ""
#define PVFORMAT_AXIS_TIMESAMPLE_DEFAULT ""
#define PVFORMAT_AXIS_TAG_DEFAULT ""
#define PVFORMAT_AXES_COMBINATION_DEFAULT ""

// Filters properties' name
#define PVFORMAT_FILTER_TYPE_STR "type"
#define PVFORMAT_FILTER_NAME_STR "name"

// Filters properties' default value
#define PVFORMAT_FILTER_TYPE_DEFAULT ""
#define PVFORMAT_FILTER_NAME_DEFAULT ""

// XML format tags
#define PVFORMAT_XML_TAG_FIELD_STR "field"
#define PVFORMAT_XML_TAG_AXIS_STR "axis"
#define PVFORMAT_XML_TAG_SPLITTER_STR "splitter"
#define PVFORMAT_XML_TAG_FILTER_STR "filter"
#define PVFORMAT_XML_TAG_AXES_COMBINATION_STR "axes-combination"

}

#endif
