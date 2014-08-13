/**
 * \file PVConfig.h
 *
 * Copyright (C) Picviz Labs 2012-2014
 */

#ifndef PVCORE_PVCONFIG_H
#define PVCORE_PVCONFIG_H

#include <QString>

namespace PVCore
{

class PVConfig
{

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

private:
	// private stuffs for a singleton declaration
	PVConfig();
	PVConfig(const PVConfig&);
	PVConfig &operator=(const PVConfig&);

private:
	static PVConfig* _pvconfig;
};

}

#endif /* PVCORE_PVCONFIG_H */
