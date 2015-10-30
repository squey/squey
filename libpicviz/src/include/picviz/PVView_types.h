/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PICVIZ_PVVIEW_TYPES_H
#define PICVIZ_PVVIEW_TYPES_H

#include <pvkernel/core/PVSharedPointer.h>
#include <pvkernel/core/PVWeakPointer.h>

namespace Picviz {

class PVView;
typedef PVCore::PVSharedPtr<PVView> PVView_sp;
typedef PVCore::PVWeakPtr<PVView>   PVView_wp;


}

#endif
