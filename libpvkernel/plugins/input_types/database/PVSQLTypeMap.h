#ifndef PVSQLTYPEMAP_FILE_H
#define PVSQLTYPEMAP_FILE_H

#include <boost/shared_ptr.hpp>
#include <QString>
#include <QHash>

namespace PVRush {

typedef QHash<int, QString> map_type;

class PVSQLTypeMap
{
public:
	typedef boost::shared_ptr<PVSQLTypeMap> p_type;
public:
	static p_type get_map(QString const& driver);
public:
	virtual QString map(int type) const = 0;
	virtual QString map_picviz(int type) const = 0;
};

typedef PVSQLTypeMap::p_type PVSQLTypeMap_p;

class PVSQLTypeMapMysql: public PVSQLTypeMap
{
public:
	QString map(int type) const;
	QString map_picviz(int type) const;
};

class PVSQLTypeMapODBC: public PVSQLTypeMap
{
public:
	QString map(int type) const { return "unknown"; }
	QString map_picviz(int type) const { return "enum"; }
};

class PVSQLTypeMapSQLite: public PVSQLTypeMap
{
public:
	QString map(int type) const { return "unknown"; }
	QString map_picviz(int type) const { return "enum"; }
};

}

#endif
