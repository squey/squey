/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2015-2015
 */

#ifndef PVOPCUAPRESETS_H
#define PVOPCUAPRESETS_H

#include "../../common/opcua/PVOpcUaInfos.h"

#include <QString>
#include <QSettings>

#include <map>

namespace PVRush
{

class PVOpcUaPresets
{
  public:
	static const char PV_SETTINGS_INPUT[];

  private:
	PVOpcUaPresets();
	~PVOpcUaPresets();

  public:
	typedef unsigned int id_t;
	typedef std::map<id_t, QString> list_id_names_t;

  public:
	static PVOpcUaPresets& get();

  public:
	id_t add(QString const& name,
	         PVOpcUaInfos const& infos,
	         QString const& query,
	         QString const& query_type);
	void set(id_t id,
	         PVOpcUaInfos const& infos,
	         QString const& query,
	         QString const& query_type,
	         QString const& name = QString());
	list_id_names_t list_id_names();
	bool get(id_t id, PVOpcUaInfos& infos, QString& query, QString& query_type);
	void rm(id_t id);

  protected:
	QSettings _settings;
};
} // namespace PVRush

#endif
