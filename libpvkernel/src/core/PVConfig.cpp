/**
 * \file PVConfig.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvkernel/core/PVConfig.h>

PVCore::PVConfig* PVCore::PVConfig::_pvconfig = nullptr;
const QString PVCore::PVConfig::_config_dir = QDir::home().absolutePath() + PICVIZ_PATH_SEPARATOR_CHAR + ".picviz";
const QString PVCore::PVConfig::_lists_folder = "lists";

void PVCore::PVConfig::init_dirs()
{
	QDir dir;
	dir.mkdir(_config_dir);
	dir.cd(_config_dir);
	dir.mkdir(get_lists_dir());
	dir.cd(get_lists_dir());
	dir.mkdir("blacklist");
	dir.mkdir("whitelist");
	dir.mkdir("greylist");
}
