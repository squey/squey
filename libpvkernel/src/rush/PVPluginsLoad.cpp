/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <QCoreApplication>
#include <QDir>

#include <stdlib.h>

int PVRush::PVPluginsLoad::load_all_plugins()
{
	int ret = 0;
	ret += load_input_type_plugins();
	ret += load_source_plugins();

	return ret;
}

int PVRush::PVPluginsLoad::load_input_type_plugins()
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(get_input_type_dir(), INPUT_TYPE_PREFIX);
	if (ret == 0) {
		PVLOG_WARN("No input plugin have been loaded !\n");
	}
	else {
		PVLOG_INFO("%d input plugins have been loaded.\n", ret);
	}
	return ret;
}

int PVRush::PVPluginsLoad::load_source_plugins()
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(get_source_dir(), SOURCE_PREFIX);
	if (ret == 0) {
		PVLOG_WARN("No source plugin have been loaded !\n");
	}
	else {
		PVLOG_INFO("%d source plugins have been loaded.\n", ret);
	}
	return ret;
}

QString PVRush::PVPluginsLoad::get_input_type_dir()
{
	QString pluginsdirs = QString(getenv("PVRUSH_INPUTTYPE_DIR"));
	if (pluginsdirs.isEmpty()) {
		pluginsdirs = QCoreApplication::applicationDirPath() + QDir::separator() + PVRUSH_INPUTTYPE_DIR;
	}

	return pluginsdirs;
}

QString PVRush::PVPluginsLoad::get_source_dir()
{
	QString pluginsdirs = QString(getenv("PVRUSH_SOURCE_DIR"));
	if (pluginsdirs.isEmpty()) {
		pluginsdirs = QCoreApplication::applicationDirPath() + QDir::separator() + PVRUSH_SOURCE_DIR;
	}

	return pluginsdirs;
}
