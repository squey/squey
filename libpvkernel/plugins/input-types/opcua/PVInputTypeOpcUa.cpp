/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

#include "PVInputTypeOpcUa.h"

#include "PVOpcUaParamsWidget.h"

bool PVRush::PVInputTypeOpcUa::createWidget(hash_formats& formats,
                                          list_inputs& inputs,
                                          QString& format,
                                          PVCore::PVArgumentList& /*args_ext*/,
                                          QWidget* parent) const
{
	connect_parent(parent);
	std::unique_ptr<PVOpcUaParamsWidget> params(new PVOpcUaParamsWidget(this, formats, parent));
	if (params->exec() == QDialog::Rejected) {
		return false;
	}

	PVOpcUaQuery* query = new PVOpcUaQuery(params->get_query());

	PVInputDescription_p ind(query);
	inputs.push_back(ind);

	if (params->is_format_custom()) {
		PVRush::PVFormat custom_format(params->get_custom_format().documentElement());
		formats["custom"] = std::move(custom_format);
		format = "custom";
	} else {
		format = params->get_format_path();
	}

	return true;
}

QString PVRush::PVInputTypeOpcUa::name() const
{
	return QString("opcua");
}

QString PVRush::PVInputTypeOpcUa::human_name() const
{
	return QString("OpcUa import plugin");
}

QString PVRush::PVInputTypeOpcUa::human_name_serialize() const
{
	return QString("OpcUa");
}

QString PVRush::PVInputTypeOpcUa::internal_name() const
{
	return QString("08-opcua");
}

QString PVRush::PVInputTypeOpcUa::menu_input_name() const
{
	return QString("OPC UA...");
}

QString PVRush::PVInputTypeOpcUa::tab_name_of_inputs(list_inputs const& in) const
{
	PVInputDescription_p query = in[0];
	return query->human_name();
}

bool PVRush::PVInputTypeOpcUa::get_custom_formats(PVInputDescription_p /*in*/,
                                                  hash_formats& /*formats*/) const
{
	return false;
}

QKeySequence PVRush::PVInputTypeOpcUa::menu_shortcut() const
{
	return QKeySequence();
}
