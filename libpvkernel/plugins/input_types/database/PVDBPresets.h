#ifndef PVDBPRESETS_H
#define PVDBPRESETS_H

#include "../../common/database/PVDBInfos.h"

#include <QString>
#include <QSettings>

#include <map>

namespace PVRush {

class PVDBPresets
{
private:
	PVDBPresets();
	~PVDBPresets();

public:
	typedef unsigned int id_t;
	typedef std::map<id_t, QString> list_id_names_t;

public:
	static PVDBPresets& get();

public:
	id_t add(QString const& name, PVDBInfos const& infos, QString const& query);
	void set(id_t id, PVDBInfos const& infos, QString const& query);
	list_id_names_t list_id_names();
	bool get(id_t id, PVDBInfos& infos, QString& query);

public:
	PVDBInfos get_dbinfos();
	QString get_query();

protected:
	QSettings _settings;
};

}

#endif
