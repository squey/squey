/**
 * \file PVSplunkPresets.cpp
 *
 * Copyright (C) Picviz Labs 2015
 */

#include <pvkernel/core/general.h>
#include <QSettings>
#include <QStringList>
#include "PVSplunkPresets.h"

const char PVRush::PVSplunkPresets::PV_SETTINGS_INPUT[] = "plugins/input_type/splunk/";

PVRush::PVSplunkPresets::PVSplunkPresets():
	_settings(QSettings::UserScope, PICVIZ_ORGANISATION, PICVIZ_APPLICATIONNAME)
{
	_settings.beginGroup(QString(PV_SETTINGS_INPUT) + "presets");
}

PVRush::PVSplunkPresets::~PVSplunkPresets()
{
	_settings.endGroup();
}

PVRush::PVSplunkPresets& PVRush::PVSplunkPresets::get()
{
	static PVSplunkPresets obj;
	return obj;
}


PVRush::PVSplunkPresets::id_t PVRush::PVSplunkPresets::add(
	QString const& name,
	PVSplunkInfos const& infos,
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

void PVRush::PVSplunkPresets::set(
	id_t id, PVSplunkInfos const& infos,
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
	_settings.setValue("splunk_index", infos.get_splunk_index());
	_settings.setValue("splunk_host", infos.get_splunk_host());
	_settings.setValue("splunk_sourcetype", infos.get_splunk_sourcetype());
	_settings.setValue("query", query);
	_settings.setValue("query_type", query_type);
	_settings.endGroup();
}

bool PVRush::PVSplunkPresets::get(id_t id, PVSplunkInfos& infos, QString& query, QString& query_type)
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
		infos.set_splunk_index(_settings.value("splunk_index", "").toString());
		infos.set_splunk_host(_settings.value("splunk_host", "").toString());
		infos.set_splunk_sourcetype(_settings.value("splunk_sourcetype", "").toString());
		query = _settings.value("query", "").toString();
		query_type = _settings.value("query_type", "").toString();
		ret = true;
	}
	_settings.endGroup();
	return ret;
}

PVRush::PVSplunkPresets::list_id_names_t PVRush::PVSplunkPresets::list_id_names()
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

void PVRush::PVSplunkPresets::rm(id_t id)
{
	_settings.beginGroup(QString::number(id));
	_settings.remove("");
	_settings.endGroup();
}
