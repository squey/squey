/**
 * \file PVConfig.cpp
 *
 * Copyright (C) Picviz Labs 2012-1024
 */

#include <pvbase/general.h>

#include <pvkernel/core/PVConfig.h>

#include <QDir>

PVCore::PVConfig* PVCore::PVConfig::_pvconfig = nullptr;

static const QString _config_dir = QDir::homePath() + QDir::separator() + PICVIZ_CONFDIR;
static const QString _lists_folder = "lists";

/*****************************************************************************
 * PVCore::PVConfig::PVConfig
 *****************************************************************************/

PVCore::PVConfig::PVConfig()
{
	// initialization of white/grey/black lists' directories
	QDir dir;
	dir.mkdir(_config_dir);
	dir.cd(_config_dir);
	dir.mkdir(get_lists_dir());
	dir.cd(get_lists_dir());
	dir.mkdir("blacklist");
	dir.mkdir("whitelist");
	dir.mkdir("greylist");
}

/*****************************************************************************
 * PVCore::PVConfig::get
 *****************************************************************************/

PVCore::PVConfig& PVCore::PVConfig::get()
{
	if (_pvconfig == nullptr) {
		_pvconfig = new PVConfig();
	}
	return *_pvconfig;
}

/*****************************************************************************
 * PVCore::PVConfig::get_lists_dir
 *****************************************************************************/

QString PVCore::PVConfig::get_lists_dir() const
{
	return _config_dir + QDir::separator() + _lists_folder;
}
