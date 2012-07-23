/**
 * \file PVMandatoryMappingFilter.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <picviz/PVMandatoryMappingFilter.h>
#include <QVector>
#include <QString>

Picviz::PVMandatoryMappingFilter::PVMandatoryMappingFilter()
{
	_mandatory_params = NULL;
}

int Picviz::PVMandatoryMappingFilter::operator()(mandatory_param_list_values const& values)
{
	const PVRush::PVNraw::const_trans_nraw_table_line* str_nraw = values.first;
	float* mapped_values = values.second;

	init_from_first(mandatory_param_value(str_nraw->at(0).get_qstr(), mapped_values[0]));
	for (PVCol i = 0; i < str_nraw->size(); i++) {
		operator()(mandatory_param_value(str_nraw->at(i).get_qstr(), mapped_values[i]));
	}

	return 0;
}

int Picviz::PVMandatoryMappingFilter::operator()(mandatory_param_value const& /*value*/)
{
	PVLOG_WARN("In default mandatory mapping filter, nothing is done !\n");
	return 0;
}

void Picviz::PVMandatoryMappingFilter::init_from_first(mandatory_param_value const& /*value*/)
{
}

void Picviz::PVMandatoryMappingFilter::set_dest_params(mandatory_param_map& params)
{
	_mandatory_params = &params;
}
