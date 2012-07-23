/**
 * \file PVInputTypeDatabase.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVInputTypeDatabase.h"
#include "PVDatabaseParamsWidget.h"

#include "../../common/database/PVDBInfos.h"

PVRush::PVInputTypeDatabase::PVInputTypeDatabase() :
	PVInputTypeDesc<PVDBQuery>(),
	_is_custom_format(false)
{
}

bool PVRush::PVInputTypeDatabase::createWidget(hash_formats const& formats, hash_formats& new_formats, list_inputs &inputs, QString& format, PVCore::PVArgumentList& args_ext, QWidget* parent) const
{
	connect_parent(parent);
	PVDatabaseParamsWidget* params = new PVDatabaseParamsWidget(this, formats, parent);
	if (params->exec() == QDialog::Rejected) {
		return false;
	}

	PVDBInfos infos;
	params->get_dbinfos(infos);
	PVDBServ_p serv(new PVDBServ(infos));
	PVInputDescription_p query(new PVDBQuery(serv, params->get_query()));

	inputs.push_back(query);

	if (params->is_format_custom()) {
		PVRush::PVFormat custom_format;
		custom_format.populate_from_xml(params->get_custom_format().documentElement());
		new_formats["custom"] = custom_format;
		format = "custom";
	}
	else {
		format = params->get_existing_format();
	}

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

QString PVRush::PVInputTypeDatabase::human_name_serialize() const
{
	return QString("Databases");
}

QString PVRush::PVInputTypeDatabase::menu_input_name() const
{
	return QString("Import from a database...");
}

QString PVRush::PVInputTypeDatabase::tab_name_of_inputs(list_inputs const& in) const
{
	PVInputDescription_p query = in[0];
	return query->human_name();
}

bool PVRush::PVInputTypeDatabase::get_custom_formats(input_type /*in*/, hash_formats& /*formats*/) const
{
	return false;
}

QKeySequence PVRush::PVInputTypeDatabase::menu_shortcut() const
{
	return QKeySequence();
}
