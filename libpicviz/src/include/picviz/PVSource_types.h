/**
 * \file PVSource_types.h
 *
 * Copyright (C) Picviz Labs 2010-2012
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
