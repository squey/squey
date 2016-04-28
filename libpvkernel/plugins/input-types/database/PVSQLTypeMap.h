/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVSQLTYPEMAP_FILE_H
#define PVSQLTYPEMAP_FILE_H

#include <memory>

#include <QString>
#include <QHash>

namespace PVRush
{

typedef QHash<int, QString> map_type;

class PVSQLTypeMap
{
  public:
	typedef std::shared_ptr<PVSQLTypeMap> p_type;

  public:
	static p_type get_map(QString const& driver);

  public:
	virtual QString map(int type) const = 0;
	virtual QString map_inendi(int type) const = 0;
};

typedef PVSQLTypeMap::p_type PVSQLTypeMap_p;

class PVSQLTypeMapMysql : public PVSQLTypeMap
{
  public:
	QString map(int type) const;
	QString map_inendi(int type) const;
};

class PVSQLTypeMapODBC : public PVSQLTypeMap
{
  public:
	QString map(int /*type*/) const { return "unknown"; }
	QString map_inendi(int /*type*/) const { return "enum"; }
};

class PVSQLTypeMapSQLite : public PVSQLTypeMap
{
  public:
	QString map(int /*type*/) const { return "unknown"; }
	QString map_inendi(int /*type*/) const { return "enum"; }
};
}

#endif
