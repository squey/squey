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
