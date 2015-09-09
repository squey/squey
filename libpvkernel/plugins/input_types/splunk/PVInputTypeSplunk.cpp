/**
 * \file PVInputTypeSplunk.cpp
 *
 * Copyright (C) Picviz Labs 2015
 */

#include "PVInputTypeSplunk.h"

#include "PVSplunkParamsWidget.h"

#include "../../common/splunk/PVSplunkInfos.h"

PVRush::PVInputTypeSplunk::PVInputTypeSplunk() :
	PVInputTypeDesc<PVSplunkQuery>(),
	_is_custom_format(false)
{
}

bool PVRush::PVInputTypeSplunk::createWidget(hash_formats const& formats, hash_formats& /*new_formats*/, list_inputs &inputs, QString& format, PVCore::PVArgumentList& /*args_ext*/, QWidget* parent) const
{
	connect_parent(parent);
	PVSplunkParamsWidget* params = new PVSplunkParamsWidget(this, formats, parent);
	if (params->exec() == QDialog::Rejected) {
		return false;
	}

	PVSplunkQuery* query = new PVSplunkQuery(params->get_query());

	PVInputDescription_p ind(query);
	inputs.push_back(ind);

	format = PICVIZ_BROWSE_FORMAT_STR;

	return true;
}

PVRush::PVInputTypeSplunk::~PVInputTypeSplunk()
{
}


QString PVRush::PVInputTypeSplunk::name() const
{
	return QString("splunk");
}

QString PVRush::PVInputTypeSplunk::human_name() const
{
	return QString("Splunk import plugin");
}

QString PVRush::PVInputTypeSplunk::human_name_serialize() const
{
	return QString("Splunk");
}

QString PVRush::PVInputTypeSplunk::internal_name() const
{
	return QString("05-splunk");
}

QString PVRush::PVInputTypeSplunk::menu_input_name() const
{
	return QString("Splunk...");
}

QString PVRush::PVInputTypeSplunk::tab_name_of_inputs(list_inputs const& in) const
{
	PVInputDescription_p query = in[0];
	return query->human_name();
}

bool PVRush::PVInputTypeSplunk::get_custom_formats(PVInputDescription_p /*in*/, hash_formats& /*formats*/) const
{
	return false;
}

QKeySequence PVRush::PVInputTypeSplunk::menu_shortcut() const
{
	return QKeySequence();
}
