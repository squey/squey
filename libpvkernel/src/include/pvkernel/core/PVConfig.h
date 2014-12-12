/**
 * \file PVConfig.h
 *
 * Copyright (C) Picviz Labs 2012-2014
 */

#ifndef PVCORE_PVCONFIG_H
#define PVCORE_PVCONFIG_H

#include <QString>
#include <QSettings>

#include <memory>

class QSettings;

namespace PVCore
{

class PVConfig
{
	typedef std::shared_ptr<PVConfig> PVConfig_p;

public:
	/**
	 * return a singleton on the global configuration accessor
	 */
	static PVConfig& get();

public:
	/**
	 * return the path to the white/grey/black lists
	 */
	QString get_lists_dir() const;

	/**
	 * return a reference on the application's settings
	 */
	QSettings& config() const;

	static QString username();

private:
	// private stuffs for a singleton declaration
	PVConfig();
	PVConfig(const PVConfig&);
	PVConfig &operator=(const PVConfig&);

public:
	// public destructor to permit its destruction
	~PVConfig();

private:
	static PVConfig_p  _pvconfig;
	QSettings         *_config;
	QString            _username;
};

}

#endif /* PVCORE_PVCONFIG_H */
