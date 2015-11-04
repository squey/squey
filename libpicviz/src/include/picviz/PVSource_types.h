/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PICVIZ_PVSOURCE_TYPES_H
#define PICVIZ_PVSOURCE_TYPES_H

#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/core/PVWeakPointer.h>

namespace Picviz {

class PVSource;
typedef PVCore::PVDataTreeAutoShared<PVSource> PVSource_p;
typedef PVCore::PVSharedPtr<PVSource> PVSource_sp;
typedef PVCore::PVWeakPtr<PVSource>   PVSource_wp;
}


#endif
