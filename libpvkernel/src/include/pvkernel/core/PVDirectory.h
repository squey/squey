/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVDIRECTORY_H
#define PVCORE_PVDIRECTORY_H

#include <pvkernel/core/general.h>
#include <QString>
#include <QDir>

namespace PVCore
{

class PVDirectory
{
  public:
	static bool remove_rec(QString const& dirName);
	static QString temp_dir(QString const& pattern);
	static QString temp_dir(QDir const& directory, QString const& pattern);
};
}

#endif
