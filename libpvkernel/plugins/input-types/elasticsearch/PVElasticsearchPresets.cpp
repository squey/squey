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

#include <pvbase/general.h>
#include <pvkernel/core/PVConfig.h>
#include <QSettings>
#include <QStringList>
#include "PVElasticsearchPresets.h"

const char PVRush::PVElasticsearchPresets::PV_SETTINGS_INPUT[] =
    "plugins/input_type/elasticsearch/";

PVRush::PVElasticsearchPresets::PVElasticsearchPresets()
    : _settings(QString::fromStdString(PVCore::PVConfig::user_dir()) + QDir::separator() +
                    PVCore::PVConfig::PRESETS_FILENAME,
                QSettings::IniFormat)
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

PVRush::PVElasticsearchPresets::id_t
PVRush::PVElasticsearchPresets::add(QString const& name,
                                    PVElasticsearchInfos const& infos,
                                    QString const& query,
                                    QString const& query_type)
{
	QStringList grps = _settings.childGroups();
	id_t max = 0;
	for (auto & grp : grps) {
		id_t id = grp.toUInt();
		if (id > max) {
			max = id;
		}
	}

	id_t new_id = max + 1;
	set(new_id, infos, query, query_type, name);
	return new_id;
}

void PVRush::PVElasticsearchPresets::set(id_t id,
                                         PVElasticsearchInfos const& infos,
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

bool PVRush::PVElasticsearchPresets::get(id_t id,
                                         PVElasticsearchInfos& infos,
                                         QString& query,
                                         QString& query_type)
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

PVRush::PVElasticsearchPresets::list_id_names_t PVRush::PVElasticsearchPresets::list_id_names()
{
	list_id_names_t ret;
	QStringList grps = _settings.childGroups();
	for (auto & grp : grps) {
		id_t id = grp.toUInt();
		QString name = _settings.value(grp + "/name", "").toString();
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
