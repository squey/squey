/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2019
 */

#include <pvbase/general.h>

#include "PVInputTypeERF.h"
#include "PVERFParamsWidget.h"

bool PVRush::PVInputTypeERF::createWidget(hash_formats& formats,
                                          list_inputs& inputs,
                                          QString& format,
                                          PVCore::PVArgumentList& /*args_ext*/,
                                          QWidget* parent) const
{
	PVERFParamsWidget* params = new PVERFParamsWidget(this, parent);
	if (params->exec() == QDialog::Rejected) {
		return false;
	}

	PVRush::PVERFDescription* desc =
	    new PVRush::PVERFDescription(params->path(), params->get_selected_nodes());

	PVInputDescription_p ind(desc);
	inputs.push_back(ind);

	const std::vector<QDomDocument> custom_formats = params->get_formats();
	for (size_t i = 0; i < custom_formats.size(); i++) {
		PVRush::PVFormat custom_format(custom_formats[i].documentElement());
		formats[QString("custom") + QString::number(i)] = std::move(custom_format);
	}
	format = "custom";

	return true;
}

QString PVRush::PVInputTypeERF::name() const
{
	return QString("erf");
}

QString PVRush::PVInputTypeERF::human_name() const
{
	return QString("ERF import plugin");
}

QString PVRush::PVInputTypeERF::human_name_serialize() const
{
	return QString("ERF");
}

QString PVRush::PVInputTypeERF::internal_name() const
{
	return QString("07-elasticsearch");
}

QString PVRush::PVInputTypeERF::menu_input_name() const
{
	return QString("ERF...");
}

QString PVRush::PVInputTypeERF::tab_name_of_inputs(list_inputs const& in) const
{
	PVInputDescription_p query = in[0];
	return query->human_name();
}

bool PVRush::PVInputTypeERF::get_custom_formats(PVInputDescription_p /*in*/,
                                                hash_formats& /*formats*/) const
{
	return false;
}

QKeySequence PVRush::PVInputTypeERF::menu_shortcut() const
{
	return QKeySequence();
}
