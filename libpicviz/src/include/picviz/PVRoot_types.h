/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PICVIZ_PVROOT_TYPES_H
#define PICVIZ_PVROOT_TYPES_H

#include <pvkernel/core/PVSharedPointer.h>
#include <pvkernel/core/PVWeakPointer.h>

namespace Picviz {

class PVRoot;
typedef PVCore::PVSharedPtr<PVRoot> PVRoot_sp;
typedef PVCore::PVWeakPtr<PVRoot>   PVRoot_wp;

}

#endif
