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

#include "PVDBInfos.h"
#include "PVDBQuery.h"

#include <QSqlRecord>
#include <QSqlField>
#include <memory>

#include <pvkernel/rush/PVXmlTreeNodeDom.h>
#include "PVSQLTypeMap.h"

#include <pvkernel/core/PVRecentItemsManager.h>

PVRush::PVDBQuery::PVDBQuery()
{
}

PVRush::PVDBQuery::PVDBQuery(PVDBServ_p infos) : _infos(infos)
{
}

PVRush::PVDBQuery::PVDBQuery(PVDBServ_p infos, QString const& query) : _infos(infos)
{
	set_query(query);
}

PVRush::PVDBQuery::~PVDBQuery()
{
}

bool PVRush::PVDBQuery::operator==(const PVInputDescription& other) const
{
	auto const* other_query = dynamic_cast<PVDBQuery const*>(&other);
	if (not other_query) {
		return false;
	}

	return _infos->get_type() == other_query->_infos->get_type() &&
	       _infos->get_host() == other_query->_infos->get_host() &&
	       _infos->get_options() == other_query->_infos->get_options() &&
	       _infos->get_dbname() == other_query->_infos->get_dbname() &&
	       _infos->get_port() == other_query->_infos->get_port();
}

QSqlQuery PVRush::PVDBQuery::to_query(chunk_index /*start*/, chunk_index /*nelts*/) const
{
	QString querystr = _query /* + QString(" LIMIT %1 OFFSET %2").arg(nelts).arg(start)*/;
	QSqlQuery query(_infos->to_database());
	query.prepare(querystr);
	return query;
}

bool PVRush::PVDBQuery::connect_serv()
{
	return _infos->connect();
}

QDomDocument PVRush::PVDBQuery::get_format_from_db_schema() const
{
	QSqlQuery query = to_query(0, 1);
	query.exec();
	QSqlRecord record = query.record();

	QDomDocument format_doc;
	PVXmlTreeNodeDom* format_root = PVRush::PVXmlTreeNodeDom::new_format(format_doc);

	PVSQLTypeMap_p type_map = PVSQLTypeMap::get_map(_infos->get_type());
	for (int i = 0; i < record.count(); i++) {
		QSqlField field = record.field(i);
		QString type = type_map->map_inendi(field.typeID());
		format_root->addOneField(field.name(), type);
	}

	return format_doc;
}

QList<QString> PVRush::PVDBQuery::get_db_types() const
{
	QSqlQuery query = to_query(0, 1);
	query.exec();
	QSqlRecord record = query.record();

	QList<QString> types;

	PVSQLTypeMap_p type_map = PVSQLTypeMap::get_map(_infos->get_type());
	for (int i = 0; i < record.count(); i++) {
		types.append(type_map->map(record.field(i).typeID()));
	}

	return types;
}

QString PVRush::PVDBQuery::last_error_serv()
{
	return _infos->last_error();
}

QString PVRush::PVDBQuery::human_name() const
{
	return _infos->database_name();
}

void PVRush::PVDBQuery::serialize_write(PVCore::PVSerializeObject& so) const
{
	so.set_current_status("Saving database information...");
	so.attribute_write("query", _query);
	_infos->serialize_write(*so.create_object("server"));
}

std::unique_ptr<PVRush::PVInputDescription>
PVRush::PVDBQuery::serialize_read(PVCore::PVSerializeObject& so)
{
	so.set_current_status("Loading database information...");
	auto query = so.attribute_read<QString>("query");
	PVDBInfos infos = PVDBInfos::serialize_read(*so.create_object("server"));
	return std::make_unique<PVDBQuery>(
	    std::make_shared<PVDBServ>(infos), query);
}

void PVRush::PVDBQuery::save_to_qsettings(QSettings& settings) const
{
	settings.setValue("type", _infos->get_type());
	settings.setValue("host", _infos->get_host());
	settings.setValue("options", _infos->get_options());
	settings.setValue("dbname", _infos->get_dbname());
	settings.setValue("port", _infos->get_port());
}

std::unique_ptr<PVRush::PVInputDescription>
PVRush::PVDBQuery::load_from_string(std::vector<std::string> const&, bool /* multi_inputs */)
{
	throw PVRush::BadInputDescription("Incomplete input for DBQuery");
}

std::vector<std::string> PVRush::PVDBQuery::desc_from_qsetting(QSettings const&)
{
	throw PVRush::BadInputDescription("Incomplete input for DBQuery");
}
