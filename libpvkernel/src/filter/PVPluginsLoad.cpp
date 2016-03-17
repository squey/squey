/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/filter/PVPluginsLoad.h>

#include <QString>
#include <iostream>

int PVFilter::PVPluginsLoad::load_all_plugins()
{
	return load_normalize_plugins();
}

int PVFilter::PVPluginsLoad::load_normalize_plugins()
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(QString::fromStdString(get_normalize_dir()), NORMALIZE_FILTER_PREFIX);
	if (ret == 0) {
		PVLOG_WARN("No normalization plugin have been loaded !\n");
	}
	else {
		PVLOG_INFO("%d normalization plugins have been loaded.\n", ret);
	}
	return ret;
}

std::string PVFilter::PVPluginsLoad::get_normalize_dir()
{
	const char* path = std::getenv("PVKERNEL_PLUGIN_PATH");
	if (path) {
		return std::string(path) + "/normalize-filters";
	}
	return std::string(PVKERNEL_PLUGIN_PATH) + "/normalize-filters";
}
