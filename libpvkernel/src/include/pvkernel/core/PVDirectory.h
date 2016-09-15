/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVDIRECTORY_H
#define PVCORE_PVDIRECTORY_H

#include <QDir>
#include <QString>

namespace PVCore
{

namespace PVDirectory
{
bool remove_rec(QString const& dirName);
QString temp_dir(QString const& pattern);
QString temp_dir(QDir const& directory, QString const& pattern);
} // namespace PVDirectory
;
} // namespace PVCore

#endif
