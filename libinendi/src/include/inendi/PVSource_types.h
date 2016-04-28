/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVSOURCE_TYPES_H
#define INENDI_PVSOURCE_TYPES_H

#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/core/PVWeakPointer.h>

namespace Inendi
{

class PVSource;
typedef PVCore::PVSharedPtr<PVSource> PVSource_sp;
typedef PVCore::PVWeakPtr<PVSource> PVSource_wp;
}

#endif
