#include "PVInputTypeDatabase.h"
#include "PVDatabaseParamsWidget.h"

#include "../../common/database/PVDBQuery.h"
#include "../../common/database/PVDBInfos.h"

PVRush::PVInputTypeDatabase::PVInputTypeDatabase() :
	PVInputType()
{
}

bool PVRush::PVInputTypeDatabase::createWidget(hash_formats const& formats, list_inputs &inputs, QString& format, QWidget* parent) const
{
	PVDatabaseParamsWidget* params = new PVDatabaseParamsWidget(parent);
	if (params->exec() == QDialog::Rejected) {
		return false;
	}

	PVDBInfos infos;
	params->get_dbinfos(infos);
	PVDBServ_p serv(new PVDBServ(infos));
	PVDBQuery query(serv, params->get_query());

	QVariant in;
	in.setValue(query);
	inputs.push_back(in);

	format = QString(PICVIZ_AUTOMATIC_FORMAT_STR);

	return true;
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
