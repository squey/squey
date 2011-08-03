/*
 * $Id: PVSplitter.cpp 3090 2011-06-09 04:59:46Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 *
 */

#include <QByteArray>
#include <QFile>
#include <QString>
#include <QStringList>

#include <pvkernel/core/debug.h>

#include <pvkernel/core/pv_splitter.h>


PVSplitter::PVSplitter(const QString &name_str)
{
	name = QString(name_str);
}


PVSplitter::~PVSplitter()
{

}
