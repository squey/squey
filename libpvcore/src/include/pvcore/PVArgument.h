//! \file PVArgument.h
//! $Id: PVArgument.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVCORE_PVARGUMENT_H
#define PVCORE_PVARGUMENT_H

#include <pvcore/general.h>
#include <QHash>
#include <QString>
#include <QVariant>

namespace PVCore {

typedef QVariant                     PVArgument;
typedef QHash<QString,PVArgument>    PVArgumentList;

QString PVArgument_to_QString(PVArgument const& v);
PVArgument QString_to_PVArgument(QString const& v);

void dump_argument_list(PVArgumentList const& l);

}

#endif
