/**
 * \file PVElasticsearchPresets.cpp
 *
 * Copyright (C) Picviz Labs 2015
 */

#include <pvkernel/core/general.h>
#include <QSettings>
#include <QStringList>
#include "PVElasticsearchPresets.h"

const char PVRush::PVElasticsearchPresets::PV_SETTINGS_INPUT[] = "plugins/input_type/elasticsearch/";

PVRush::PVElasticsearchPresets::PVElasticsearchPresets():
	_settings(QSettings::UserScope, PICVIZ_ORGANISATION, PICVIZ_APPLICATIONNAME)
{
	_settings.beginGroup(QString(PV_SETTINGS_INPUT) + "presets");
}

PVRush::PVElasticsearchPresets::~PVElasticsearchPresets()
{
	_settings.endGroup();
}

PVRush::PVElasticsearchPresets& PVRush::PVElasticsearchPresets::get()
{
	static PVElasticsearchPresets obj;
	return obj;
}


PVRush::PVElasticsearchPresets::id_t PVRush::PVElasticsearchPresets::add(
	QString const& name,
	PVElasticsearchInfos const& infos,
	QString const& query,
	QString const& query_type
)
{
	QStringList grps = _settings.childGroups();
	id_t max = 0;
	for (int i = 0; i < grps.size(); i++) {
		id_t id = grps[i].toUInt();
		if (id > max) {
			max = id;
		}
	}

	id_t new_id = max+1;
	set(new_id, infos, query, query_type, name);
	return new_id;
}

void PVRush::PVElasticsearchPresets::set(
	id_t id, PVElasticsearchInfos const& infos,
	QString const& query,
	QString const& query_type,
	QString const& name
)
{
	QString grp = QString::number(id);
	_settings.beginGroup(grp);
	if (!name.isNull()) {
		_settings.setValue("name", name);
	}
	_settings.setValue("host", infos.get_host());
	_settings.setValue("port", infos.get_port());
	_settings.setValue("login", infos.get_login());
	_settings.setValue("password", infos.get_password());
	_settings.setValue("index", infos.get_index());
	_settings.setValue("query", query);
	_settings.setValue("query_type", query_type);
	_settings.endGroup();
}

bool PVRush::PVElasticsearchPresets::get(id_t id, PVElasticsearchInfos& infos, QString& query, QString& query_type)
{
	QString grp = QString::number(id);
	_settings.beginGroup(grp);
	bool ret = false;
	if (_settings.contains("name")) {
		infos.set_host(_settings.value("host", "").toString());
		infos.set_port(_settings.value("port", "").toUInt());
		infos.set_port((uint16_t) _settings.value("port", 0).toUInt());
		infos.set_login(_settings.value("login", "").toString());
		infos.set_password(_settings.value("password", "").toString());
		infos.set_index(_settings.value("index", "").toString());
		query = _settings.value("query", "").toString();
		query_type = _settings.value("query_type", "JSON").toString();
		ret = true;
	}
	_settings.endGroup();
	return ret;
}

PVRush::PVElasticsearchPresets::list_id_names_t PVRush::PVElasticsearchPresets::list_id_names()
{
	list_id_names_t ret;
	QStringList grps = _settings.childGroups();
	for (int i = 0; i < grps.size(); i++) {
		id_t id = grps[i].toUInt();
		QString name = _settings.value(grps[i] + "/name", "").toString();
		if (name.isEmpty()) {
			continue;
		}
		ret[id] = name;
	}
	return ret;
}

void PVRush::PVElasticsearchPresets::rm(id_t id)
{
	_settings.beginGroup(QString::number(id));
	_settings.remove("");
	_settings.endGroup();
}
