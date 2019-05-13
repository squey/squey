/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2019
 */

#include <pvbase/general.h>

#include "PVInputTypeERF.h"

bool PVRush::PVInputTypeERF::createWidget(hash_formats const& formats,
                                          hash_formats& new_formats,
                                          list_inputs& inputs,
                                          QString& format,
                                          PVCore::PVArgumentList& /*args_ext*/,
                                          QWidget* parent) const
{
#if 0
	connect_parent(parent);
	std::unique_ptr<PVERFParamsWidget> params(
	    new PVERFParamsWidget(this, formats, parent));
	if (params->exec() == QDialog::Rejected) {
		return false;
	}

	PVElasticsearchQuery* query = new PVElasticsearchQuery(params->get_query());

	PVInputDescription_p ind(query);
	inputs.push_back(ind);

	if (params->is_format_custom()) {
		PVRush::PVFormat custom_format(params->get_custom_format().documentElement());
		new_formats["custom"] = std::move(custom_format);
		format = "custom";
	} else {
		format = params->get_format_path();
	}
#endif
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
