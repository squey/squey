/**
 * \file PVConfig.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef _PVCORE_PVCONFIG_H__
#define _PVCORE_PVCONFIG_H__

#include <QDir>
#include <QString>

#include <pvbase/general.h>

namespace PVCore
{

class PVConfig
{
public:
	static PVConfig& get()
	{
		if (_pvconfig == nullptr) {
			_pvconfig = new PVConfig();
		}
		return *_pvconfig;
	}

	void init_dirs();

	QString get_lists_dir() { return _config_dir + PICVIZ_PATH_SEPARATOR_CHAR + _lists_folder; }

private:
	PVConfig() {}
	PVConfig(const PVConfig&);
	PVConfig &operator=(const PVConfig&);

private:
	static PVConfig* _pvconfig;
	static const QString _config_dir;
	static const QString _lists_folder;
};

}

#endif /* _PVCORE_PVCONFIG_H__ */
