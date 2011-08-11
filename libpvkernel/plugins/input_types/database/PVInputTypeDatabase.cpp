#include "PVInputTypeDatabase.h"
#include "../../common/database/PVDBQuery.h"
#include "../../common/database/PVDBInfos.h"

PVRush::PVInputTypeDatabase::PVInputTypeDatabase() :
	PVInputType()
{
}

bool PVRush::PVInputTypeDatabase::createWidget(hash_formats const& formats, list_inputs &inputs, QString& format, QWidget* parent) const
{
	PVDBInfos_p infos(new PVDBInfos("QMYSQL3", "127.0.0.1", 3306, "picviz", "picviz", "bigdata"));
	PVDBQuery query(infos, "select * from squid");

	QVariant in;
	in.setValue(query);
	inputs.push_back(in);

	format = QString(PICVIZ_AUTOMATIC_FORMAT_STR);

	return inputs.size() > 0;
}

PVRush::PVInputTypeDatabase::~PVInputTypeDatabase()
{
}


QString PVRush::PVInputTypeDatabase::name() const
{
	return QString("database");
}

QString PVRush::PVInputTypeDatabase::human_name() const
{
	return QString("Database import plugin");
}

QString PVRush::PVInputTypeDatabase::human_name_of_input(PVCore::PVArgument const& in) const
{
	return in.value<PVDBQuery>().human_name();
}

QString PVRush::PVInputTypeDatabase::menu_input_name() const
{
	return QString("Import from a database...");
}

QString PVRush::PVInputTypeDatabase::tab_name_of_inputs(list_inputs const& in) const
{
	return QString("TODO: database tab name");
}

bool PVRush::PVInputTypeDatabase::get_custom_formats(PVCore::PVArgument const& in, hash_formats &formats) const
{
	return false;
}

QKeySequence PVRush::PVInputTypeDatabase::menu_shortcut() const
{
	return QKeySequence();
}
