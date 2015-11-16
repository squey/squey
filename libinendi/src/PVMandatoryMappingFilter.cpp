/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVMandatoryMappingFilter.h>
#include <QVector>
#include <QString>

Inendi::PVMandatoryMappingFilter::PVMandatoryMappingFilter()
{
	_mandatory_params = NULL;
}

int Inendi::PVMandatoryMappingFilter::operator()(mandatory_param_list_values const& /*values*/)
{
	return 0;
}

void Inendi::PVMandatoryMappingFilter::set_dest_params(mandatory_param_map& params)
{
	_mandatory_params = &params;
}
