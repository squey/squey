/**
 * \file init.cpp
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#include <pvkernel/core/PVClassLibrary.h>

#include <sad/init.h>

#define PLUGIN_PREFIX "engine"

int Sad::initialize(void)
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(get_plugins_dir(), PLUGIN_PREFIX);

	return 0;
}

QString Sad::get_plugins_dir()
{
        QString pluginsdirs;

        pluginsdirs = QString(getenv("SAD_PLUGINS_DIR"));
        if (pluginsdirs.isEmpty()) {
                pluginsdirs = QString(SAD_PLUGINS_DIR);
        }

        return pluginsdirs;
}
