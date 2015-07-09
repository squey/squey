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

#define PV_SETTINGS_INPUT_ELASTICSEARCH "plugins/input_type/elasticsearch/"

namespace PVRush {

class PVElasticsearchPresets
{
private:
	PVElasticsearchPresets();
	~PVElasticsearchPresets();

public:
	typedef unsigned int id_t;
	typedef std::map<id_t, QString> list_id_names_t;

public:
	static PVElasticsearchPresets& get();

public:
	id_t add(QString const& name, PVElasticsearchInfos const& infos, QString const& query);
	void set(id_t id, PVElasticsearchInfos const& infos, QString const& query, QString const& name = QString());
	list_id_names_t list_id_names();
	bool get(id_t id, PVElasticsearchInfos& infos, QString& query);
	void rm(id_t id);

protected:
	QSettings _settings;
};

}

#endif
