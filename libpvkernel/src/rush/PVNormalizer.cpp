/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <QLibrary>
#include <QList>
#include <QString>
#include <QStringList>
#include <QRegExp>
#include <QHashIterator>
#include <QHash>
#include <QDir>
#include <QCoreApplication>

#include <stdlib.h>

#include <pvbase/general.h>
#include <pvkernel/rush/PVNormalizer.h>

/******************************************************************************
 *
 * PVRush::normalize_get_helpers_plugins_dirs
 *
 *****************************************************************************/
QStringList PVRush::normalize_get_helpers_plugins_dirs(QString helper)
{
	QString pluginsdirs(std::getenv("PVFORMAT_HELPER"));
	if (pluginsdirs.isEmpty()) {
		pluginsdirs = QString(INENDI_CONFIG);
	}
	pluginsdirs += "/normalize-helpers";

	QStringList pluginsdirs_list = pluginsdirs.split(PVCORE_DIRECTORY_SEP);
	for (int counter = 0; counter < pluginsdirs_list.count(); counter++) {
		if (pluginsdirs_list[counter].startsWith("~/")) {
			pluginsdirs_list[counter].replace(0, 1, QDir::homePath());
		}
		pluginsdirs_list[counter] =
		    pluginsdirs_list[counter] + QDir::separator() + helper + QDir::separator();
	}

	return pluginsdirs_list;
}
