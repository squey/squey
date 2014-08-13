/**
 * \file PVConfig.cpp
 *
 * Copyright (C) Picviz Labs 2012-1024
 */

#include <pvbase/general.h>

#include <pvkernel/core/PVConfig.h>

#include <QDir>
#include <QSettings>

#define GLOBAL_CONFIG_FILENAME "pvconfig.ini"
#define CONFIG_FILENAME "config.ini"

PVCore::PVConfig::PVConfig_p PVCore::PVConfig::_pvconfig;

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

	QFileInfo fi(QDir::homePath() + QDir::separator() + PICVIZ_INSPECTOR_CONFDIR + QDir::separator() + CONFIG_FILENAME);

	if (fi.exists() == false) {
		fi.dir().mkpath(fi.path());

		QFileInfo sys_fi(GLOBAL_CONFIG_FILENAME);

		if (sys_fi.exists()) {
			QFile::copy(sys_fi.filePath(), fi.filePath());
		}
	}

	_config = new QSettings(fi.filePath(), QSettings::IniFormat);
}

/*****************************************************************************
 * PVCore::PVConfig::~PVConfig
 *****************************************************************************/

PVCore::PVConfig::~PVConfig()
{
	if (_config) {
		delete _config;
		_config = nullptr;
	}
}

/*****************************************************************************
 * PVCore::PVConfig::get
 *****************************************************************************/

PVCore::PVConfig& PVCore::PVConfig::get()
{
	if (_pvconfig == nullptr) {
		_pvconfig = PVConfig_p(new PVConfig());
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

/*****************************************************************************
 * PVCore::PVConfig::config
 *****************************************************************************/

QSettings& PVCore::PVConfig::config() const
{
	return *_config;
}
