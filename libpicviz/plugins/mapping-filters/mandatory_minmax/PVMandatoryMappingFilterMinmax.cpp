/**
 * \file PVMandatoryMappingFilterMinmax.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVMandatoryMappingFilterMinmax.h"
#include <pvkernel/core/PVTBBMinMaxArray.h>

#include <QVector>
#include <QString>

Picviz::PVMandatoryMappingFilterMinmax::PVMandatoryMappingFilterMinmax() :
	PVMandatoryMappingFilter()
{
	INIT_FILTER_NOPARAM(PVMandatoryMappingFilterMinmax);
}

int Picviz::PVMandatoryMappingFilterMinmax::operator()(mandatory_param_list_values const& values)
{
	assert(_mandatory_params);
	float* mapped_values = values.second;
	const PVRush::PVNraw::const_trans_nraw_table_line* str_values = values.first;

	// Find the minimum and maximum of the mapped values thanks to TBB (using a reduction) !
	PVCore::PVTBBMinMaxArray<float> mmar(mapped_values);
	tbb::parallel_reduce(tbb::blocked_range<uint64_t>(0, str_values->size()), mmar, tbb::auto_partitioner());

	// And save the min/max values !
	mandatory_param_value ymin = mandatory_param_value(str_values->at(mmar.get_min_index()).get_qstr(), mmar.get_min_value());
	mandatory_param_value ymax = mandatory_param_value(str_values->at(mmar.get_max_index()).get_qstr(), mmar.get_max_value());

	_mandatory_params->insert(mandatory_param_map::value_type(Picviz::mandatory_ymin, ymin));
	_mandatory_params->insert(mandatory_param_map::value_type(Picviz::mandatory_ymax, ymax));

	return 0;
}

IMPL_FILTER_NOPARAM(Picviz::PVMandatoryMappingFilterMinmax)
