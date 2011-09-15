#include <pvcore/general.h>
#include "PVDBInfos.h"

#include <QSettings>

PVRush::PVDBPresets::PVDBPresets():
	_settings(QSettings::UserScope, PICVIZ_ORGANISATION, PICVIZ_APPLICATIONNAME)
{
	_settings.beginGroup("plugins/input_type/database/presets");
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


void PVRush::PVDBPresets::add(QString const& name, PVDBInfos const& infos, QString const& query)
{
	QStringList grps = childGroups();
	id_t max = 0;
	for (int i = 0; i < grps.size(); i++) {
		id_t id = grps[i].toUInt();
		if (id > max) {
			max = id;
		}
	}

	id_t new_id = max+1;
	set(new_id, infos, query);
}

void PVRush::PVDBPresets::set(id_t id, PVDBInfos const& infos, QString const& query)
{
	QString grp = QString::number(id);
	_settings.beginGroup(grp);
	_settings.setValue("name", name);
	_settings.setValue("host", infos.get_host());
	_settings.setValue("username", infos.get_username());
	_settings.setValue("password", infos.get_password());
	_settings.setValue("type", infos.get_type());
	_settings.setValue("port", infos.get_port());
	_settings.setValue("dbname", infos.get_dbname());
	_settings.setValue("query", query);
	_settings.endGroup();
}

void PVRush::PVDBPresets::get(id_t id, PVDBInfos& infos, QString& query)
{
	QString grp = QString::number(id);
	_settings.beginGroup(grp);
	infos.set_host(_settings.value("host", "").toString());
	infos.set_dbname(_settings.value("dbname", "").toString());
	infos.set_port(_settings.value("port", "").toString());
	infos.set_username(_settings.value("username", "").toString());
	infos.set_password(_settings.value("password", "").toString());
	infos.set_port((uint16_t) _settings.value("port", 0).toUInt());
	infos.set_type(_settings.value("type", "").toString());
	_query = _settings.value("query", "").toString();
	_settings.endGroup();
}

PVRush::PVDBPresets::list_id_names_t PVRush::PVDBPresets::list_id_names()
{
	list_id_names_t ret;
	QStringList grps = childGroups();
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
