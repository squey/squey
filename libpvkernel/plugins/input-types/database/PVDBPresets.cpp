//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include "PVDBPresets.h"

#include <pvbase/general.h>
#include <pvkernel/core/PVConfig.h>

#include <QSettings>
#include <QStringList>

PVRush::PVDBPresets::PVDBPresets()
    : _settings(QString::fromStdString(PVCore::PVConfig::user_dir()) + QDir::separator() +
                    PVCore::PVConfig::PRESETS_FILENAME,
                QSettings::IniFormat)
{
	_settings.beginGroup(PV_SETTINGS_INPUT_DB "presets");
}

PVRush::PVDBPresets::~PVDBPresets()
{
	_settings.endGroup();
}

PVRush::PVDBPresets& PVRush::PVDBPresets::get()
{
	static PVDBPresets obj;
	return obj;
}

PVRush::PVDBPresets::id_t
PVRush::PVDBPresets::add(QString const& name, PVDBInfos const& infos, QString const& query)
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
	set(new_id, infos, query, name);
	return new_id;
}

void PVRush::PVDBPresets::set(id_t id,
                              PVDBInfos const& infos,
                              QString const& query,
                              QString const& name)
{
	QString grp = QString::number(id);
	_settings.beginGroup(grp);
	if (!name.isNull()) {
		_settings.setValue("name", name);
	}
	_settings.setValue("host", infos.get_host());
	_settings.setValue("username", infos.get_username());
	_settings.setValue("password", infos.get_password());
	_settings.setValue("type", infos.get_type());
	_settings.setValue("port", infos.get_port());
	_settings.setValue("dbname", infos.get_dbname());
	_settings.setValue("query", query);
	_settings.endGroup();
}

bool PVRush::PVDBPresets::get(id_t id, PVDBInfos& infos, QString& query)
{
	QString grp = QString::number(id);
	_settings.beginGroup(grp);
	bool ret = false;
	if (_settings.contains("name")) {
		infos.set_host(_settings.value("host", "").toString());
		infos.set_dbname(_settings.value("dbname", "").toString());
		infos.set_port(_settings.value("port", "").toUInt());
		infos.set_username(_settings.value("username", "").toString());
		infos.set_password(_settings.value("password", "").toString());
		infos.set_port((uint16_t)_settings.value("port", 0).toUInt());
		infos.set_type(_settings.value("type", "").toString());
		query = _settings.value("query", "").toString();
		ret = true;
	}
	_settings.endGroup();
	return ret;
}

PVRush::PVDBPresets::list_id_names_t PVRush::PVDBPresets::list_id_names()
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

void PVRush::PVDBPresets::rm(id_t id)
{
	_settings.beginGroup(QString::number(id));
	_settings.remove("");
	_settings.endGroup();
}
