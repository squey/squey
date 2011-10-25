#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/filter/PVPluginsLoad.h>

#include <QStringList>
#include <stdlib.h>

int PVFilter::PVPluginsLoad::load_all_plugins()
{
	int ret = 0;
	ret += load_normalize_plugins();

	return ret;
}

int PVFilter::PVPluginsLoad::load_normalize_plugins()
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(get_normalize_dir(), NORMALIZE_FILTER_PREFIX);
	if (ret == 0) {
		PVLOG_WARN("No normalization plugin have been loaded !\n");
	}
	else {
		PVLOG_INFO("%d normalization plugins have been loaded.\n", ret);
	}
	return ret;
}

QString PVFilter::PVPluginsLoad::get_normalize_dir()
{
	QString pluginsdirs;
	QStringList pluginsdirs_list; 

	pluginsdirs = QString(getenv("PVFILTER_NORMALIZE_DIR"));
	if (pluginsdirs.isEmpty()) {
		pluginsdirs = QString(PVFILTER_NORMALIZE_DIR);
	}

	return pluginsdirs;
}
