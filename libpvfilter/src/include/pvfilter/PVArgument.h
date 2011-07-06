//! \file PVArgument.h
//! $Id: PVArgument.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVARGUMENT_H
#define PVFILTER_PVARGUMENT_H

#include <pvcore/general.h>
#include <QHash>
#include <QString>
#include <QVariant>

namespace PVFilter {

typedef QVariant                     PVArgument;
typedef QHash<QString,PVArgument>    PVArgumentList;

}

#endif
