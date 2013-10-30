#include "PVInputTypeArcsight.h"
#include "PVArcsightParamsWidget.h"

#include "../../common/arcsight/PVArcsightInfos.h"

PVRush::PVInputTypeArcsight::PVInputTypeArcsight() :
	PVInputTypeDesc<PVArcsightQuery>(),
	_is_custom_format(false)
{
}

bool PVRush::PVInputTypeArcsight::createWidget(hash_formats const& formats, hash_formats& new_formats, list_inputs &inputs, QString& format, PVCore::PVArgumentList& /*args_ext*/, QWidget* parent) const
{
	connect_parent(parent);
	PVArcsightParamsWidget* params = new PVArcsightParamsWidget(this, formats, parent);
	if (params->exec() == QDialog::Rejected) {
		return false;
	}

	PVArcsightQuery* query = new PVArcsightQuery();
	params->get_query(*query);

	PVInputDescription_p ind(query);
	inputs.push_back(ind);

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

PVRush::PVInputTypeArcsight::~PVInputTypeArcsight()
{
}


QString PVRush::PVInputTypeArcsight::name() const
{
	return QString("arcsight");
}

QString PVRush::PVInputTypeArcsight::human_name() const
{
	return QString("Arcsight import plugin");
}

QString PVRush::PVInputTypeArcsight::human_name_serialize() const
{
	return QString("Arcsight");
}

QString PVRush::PVInputTypeArcsight::menu_input_name() const
{
	return QString("Arcsight...");
}

QString PVRush::PVInputTypeArcsight::tab_name_of_inputs(list_inputs const& in) const
{
	PVInputDescription_p query = in[0];
	return query->human_name();
}

bool PVRush::PVInputTypeArcsight::get_custom_formats(PVInputDescription_p /*in*/, hash_formats& /*formats*/) const
{
	return false;
}

QKeySequence PVRush::PVInputTypeArcsight::menu_shortcut() const
{
	return QKeySequence();
}
