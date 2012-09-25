/**
 * \file PVMandatoryMappingFilterMinmax.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVMandatoryMappingFilterMinmax.h"
#include <pvkernel/core/PVTBBMinMaxArray.h>
#include <picviz/PVMapped.h>

#include <QVector>
#include <QString>

Picviz::PVMandatoryMappingFilterMinmax::PVMandatoryMappingFilterMinmax() :
	PVMandatoryMappingFilter()
{
	INIT_FILTER_NOPARAM(PVMandatoryMappingFilterMinmax);
}

struct min_max_holder
{
	template <typename T>
	static void call(Picviz::PVMapped const* mapped, Picviz::mandatory_param_list_values const& values, Picviz::mandatory_param_map* mandatory_params)
	{
		T const* mapped_values = &values.second->storage_cast<T>();

		// Find the minimum and maximum of the mapped values thanks to TBB (using a reduction) !
		PVCore::PVTBBMinMaxArray<T> mmar(mapped_values);
		tbb::parallel_reduce(tbb::blocked_range<uint64_t>(0, mapped->get_row_count()), mmar, tbb::auto_partitioner());

		// And save the min/max values !
		Picviz::PVMandatoryMappingFilter::decimal_storage_type ds;
		ds.storage_cast<T>() = mmar.get_min_value();
		Picviz::mandatory_param_value ymin(mapped->get_parent()->get_value(mmar.get_min_index(), values.first), ds);
		ds.storage_cast<T>() = mmar.get_max_value();
		Picviz::mandatory_param_value ymax(mapped->get_parent()->get_value(mmar.get_max_index(), values.first), ds);

		mandatory_params->insert(Picviz::mandatory_param_map::value_type(Picviz::mandatory_ymin, ymin));
		mandatory_params->insert(Picviz::mandatory_param_map::value_type(Picviz::mandatory_ymax, ymax));
	}
};

int Picviz::PVMandatoryMappingFilterMinmax::operator()(mandatory_param_list_values const& values)
{
	assert(_mandatory_params);
	decimal_storage_type::call_from_type<min_max_holder>(_decimal_type, _mapped, values, _mandatory_params);

	return 0;
}

IMPL_FILTER_NOPARAM(Picviz::PVMandatoryMappingFilterMinmax)
