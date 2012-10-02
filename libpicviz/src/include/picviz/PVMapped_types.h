#ifndef PICVIZ_PVMAPPED_TYPES_H
#define PICVIZ_PVMAPPED_TYPES_H

#include <pvkernel/core/PVDecimalStorage.h>
#include <pvkernel/core/PVSharedPointer.h>
#include <pvkernel/core/PVWeakPointer.h>

namespace Picviz {

class PVMapped;
typedef PVCore::PVDecimalStorage<32> mapped_decimal_storage_type;
typedef PVCore::PVWeakPtr<PVMapped> PVMapped_wp;
typedef PVCore::PVSharedPtr<PVMapped> PVMapped_sp;

}


#endif
