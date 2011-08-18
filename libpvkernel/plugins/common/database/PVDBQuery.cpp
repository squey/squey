#include "PVDBInfos.h"
#include "PVDBQuery.h"

PVRush::PVDBQuery::PVDBQuery()
{
}

PVRush::PVDBQuery::PVDBQuery(PVDBInfos_p infos):
	_infos(infos)
{
}

PVRush::PVDBQuery::PVDBQuery(PVDBInfos_p infos, QString const& query):
	_infos(infos)
{
	set_query(query);
}

QSqlQuery PVRush::PVDBQuery::to_query(chunk_index start, chunk_index nelts) const
{
	QString querystr = _query + QString(" LIMIT %1 OFFSET %2").arg(nelts).arg(start);
	QSqlQuery query(_infos->to_database());
	query.prepare(querystr);
	return query;
}

bool PVRush::PVDBQuery::connect_serv()
{
	return _infos->connect();
}

QString PVRush::PVDBQuery::last_error_serv()
{
	return _infos->last_error();
}

QString PVRush::PVDBQuery::human_name() const
{
	return _infos->database_name();
}
