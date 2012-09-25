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

int Picviz::PVMandatoryMappingFilter::operator()(mandatory_param_list_values const& /*values*/)
{
	return 0;
}

void Picviz::PVMandatoryMappingFilter::set_dest_params(mandatory_param_map& params)
{
	_mandatory_params = &params;
}
