/**
 * \file
 *
 * Copyright (C) Picviz Labs 2015
 */

#ifndef PVSplunkPRESETS_H
#define PVSplunkPRESETS_H

#include "../../common/splunk/PVSplunkInfos.h"

#include <QString>
#include <QSettings>

#include <map>

namespace PVRush {

class PVSplunkPresets
{
public:
	static const char PV_SETTINGS_INPUT[];

private:
	PVSplunkPresets();
	~PVSplunkPresets();

public:
	typedef unsigned int id_t;
	typedef std::map<id_t, QString> list_id_names_t;

public:
	static PVSplunkPresets& get();

public:
	id_t add(QString const& name, PVSplunkInfos const& infos, QString const& query, QString const& query_type);
	void set(id_t id, PVSplunkInfos const& infos, QString const& query, QString const& query_type, QString const& name = QString());
	list_id_names_t list_id_names();
	bool get(id_t id, PVSplunkInfos& infos, QString& query, QString& query_type);
	void rm(id_t id);

protected:
	QSettings _settings;
};

}

#endif
