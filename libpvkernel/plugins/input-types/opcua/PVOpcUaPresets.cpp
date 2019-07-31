/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

#include <pvbase/general.h>
#include <QSettings>
#include <QStringList>
#include "PVOpcUaPresets.h"

const char PVRush::PVOpcUaPresets::PV_SETTINGS_INPUT[] = "plugins/input_type/elasticsearch/";

PVRush::PVOpcUaPresets::PVOpcUaPresets()
    : _settings(QSettings::UserScope, INENDI_ORGANISATION, INENDI_APPLICATIONNAME)
{
	_settings.beginGroup(QString(PV_SETTINGS_INPUT) + "presets");
}

PVRush::PVOpcUaPresets::~PVOpcUaPresets()
{
	_settings.endGroup();
}

PVRush::PVOpcUaPresets& PVRush::PVOpcUaPresets::get()
{
	static PVOpcUaPresets obj;
	return obj;
}

PVRush::PVOpcUaPresets::id_t PVRush::PVOpcUaPresets::add(QString const& name,
                                                         PVOpcUaInfos const& infos,
                                                         QString const& query,
                                                         QString const& query_type)
{
	QStringList grps = _settings.childGroups();
	id_t max = 0;
	for (int i = 0; i < grps.size(); i++) {
		id_t id = grps[i].toUInt();
		if (id > max) {
			max = id;
		}
	}

	id_t new_id = max + 1;
	set(new_id, infos, query, query_type, name);
	return new_id;
}

void PVRush::PVOpcUaPresets::set(id_t id,
                                 PVOpcUaInfos const& infos,
                                 QString const& query,
                                 QString const& query_type,
                                 QString const& name)
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
	_settings.setValue("format", infos.get_format());
	_settings.setValue("is_format_custom", infos.is_format_custom());
	_settings.setValue("filter_path", infos.get_filter_path());
	_settings.endGroup();
}

bool PVRush::PVOpcUaPresets::get(id_t id, PVOpcUaInfos& infos, QString& query, QString& query_type)
{
	QString grp = QString::number(id);
	_settings.beginGroup(grp);
	bool ret = false;
	if (_settings.contains("name")) {
		infos.set_host(_settings.value("host", "").toString());
		infos.set_port(_settings.value("port", "").toUInt());
		infos.set_port((uint16_t)_settings.value("port", 0).toUInt());
		infos.set_login(_settings.value("login", "").toString());
		infos.set_password(_settings.value("password", "").toString());
		infos.set_index(_settings.value("index", "").toString());
		query = _settings.value("query", "").toString();
		query_type = _settings.value("query_type", "JSON").toString();
		infos.set_format(_settings.value("format", "").toString());
		infos.set_custom_format(_settings.value("is_format_custom", "").toString() == "true");
		infos.set_filter_path(_settings.value("filter_path", "").toString());
		ret = true;
	}
	_settings.endGroup();
	return ret;
}

PVRush::PVOpcUaPresets::list_id_names_t PVRush::PVOpcUaPresets::list_id_names()
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

void PVRush::PVOpcUaPresets::rm(id_t id)
{
	_settings.beginGroup(QString::number(id));
	_settings.remove("");
	_settings.endGroup();
}
