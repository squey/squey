/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/filter/PVPluginsLoad.h> // for NORMALIZE_FILTER_PREFIX

#include <pvkernel/core/PVClassLibrary.h> // for PVClassLibraryLibLoader
#include <pvkernel/core/PVLogger.h>       // for PVLOG_INFO, PVLOG_WARN

#include <QString>

#include <cstdlib> // for getenv
#include <string>  // for allocator, operator+, etc

int PVFilter::PVPluginsLoad::load_all_plugins()
{
	return load_normalize_plugins();
}

int PVFilter::PVPluginsLoad::load_normalize_plugins()
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(
	    QString::fromStdString(get_normalize_dir()), NORMALIZE_FILTER_PREFIX);
	if (ret == 0) {
		PVLOG_WARN("No normalization plugin have been loaded !\n");
	} else {
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
