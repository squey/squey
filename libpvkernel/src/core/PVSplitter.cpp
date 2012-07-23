/**
 * \file PVSplitter.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
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
