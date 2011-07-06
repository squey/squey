//! \file PVAxisIndexType.h
//! $Id: PVAxesIndexType.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVCORE_PVAXESINDEXTYPE_H
#define PVCORE_PVAXESINDEXTYPE_H

#include <pvcore/general.h>
#include <pvcore/types.h>

#include <vector>

#include <QMetaType>

namespace PVCore {

typedef std::vector<PVCol> PVAxesIndexType;

}

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(PVCore::PVAxesIndexType)


#endif // PVCORE_PVAXESINDEXTYPE_H
