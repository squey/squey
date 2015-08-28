/**
 * \file PVElasticsearchPresets.h
 *
 * Copyright (C) Picviz Labs 2015
 */

#ifndef PVELASTICSEARCHPRESETS_H
#define PVELASTICSEARCHPRESETS_H

#include "../../common/elasticsearch/PVElasticsearchInfos.h"

#include <QString>
#include <QSettings>

#include <map>

namespace PVRush {

class PVElasticsearchPresets
{
public:
	static const char PV_SETTINGS_INPUT[];

private:
	PVElasticsearchPresets();
	~PVElasticsearchPresets();

public:
	typedef unsigned int id_t;
	typedef std::map<id_t, QString> list_id_names_t;

public:
	static PVElasticsearchPresets& get();

public:
	id_t add(QString const& name, PVElasticsearchInfos const& infos, QString const& query, QString const& query_type);
	void set(id_t id, PVElasticsearchInfos const& infos, QString const& query, QString const& query_type, QString const& name = QString());
	list_id_names_t list_id_names();
	bool get(id_t id, PVElasticsearchInfos& infos, QString& query, QString& query_type);
	void rm(id_t id);

protected:
	QSettings _settings;
};

}

#endif
