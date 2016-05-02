/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVMAPPED_TYPES_H
#define INENDI_PVMAPPED_TYPES_H

#include <pvkernel/core/PVDecimalStorage.h>
#include <pvkernel/core/PVSharedPointer.h>
#include <pvkernel/core/PVWeakPointer.h>

namespace Inendi
{

class PVMapped;
typedef PVCore::PVDecimalStorage<32> mapped_decimal_storage_type;
typedef PVCore::PVSharedPtr<PVMapped> PVMapped_sp;
}

#endif
