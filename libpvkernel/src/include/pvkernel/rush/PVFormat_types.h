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

#define PVFORMAT_AXES_COMBINATION "axes-combination"

}

#endif
