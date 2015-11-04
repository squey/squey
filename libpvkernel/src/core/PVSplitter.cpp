/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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
