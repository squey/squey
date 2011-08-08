//! \file PVCore::PVFieldsMappingFilter.cpp
//! $Id: PVFieldsMappingFilter.cpp 3221 2011-06-30 11:45:19Z aguinet $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsMappingFilter.h>
#include <list>

/******************************************************************************
 *
 * PVFilter::PVCore::PVFieldsMappingFilter::PVCore::PVFieldsMappingFilter
 *
 *****************************************************************************/
PVFilter::PVFieldsMappingFilter::PVFieldsMappingFilter(map_filters const& mfilters) :
	_mfilters(mfilters)
{
}

/******************************************************************************
 *
 * PVFilter::PVFieldsMappingFilter::operator
 *
 *****************************************************************************/
PVCore::list_fields& PVFilter::PVFieldsMappingFilter::many_to_many(PVCore::list_fields &fields)
{
	//TODO: this *can* be optimised !
	if (fields.size() == 0)
		return fields;
	map_filters::const_iterator it,ite;
	ite = _mfilters.end();
	PVCore::list_fields tmp_fields;
	for (it = _mfilters.begin(); it != ite; it++) {
		list_indexes const& indx = (*it).first;
		PVFieldsBaseFilter_f f = (*it).second;

		tmp_fields.clear();
		list_indexes::const_iterator it_ind;
		for (it_ind = indx.begin(); it_ind != indx.end(); it_ind++)
		{
			PVCore::list_fields::iterator it_curf = fields.begin();
			std::advance(it_curf, *it_ind);
			if (it_curf == fields.end()) {
				PVLOG_WARN("(PVFieldsMappingFilter) element hasn't enough field to apply mapping (index %d requested, %d fields available). Ignoring element...\n", *it_ind, fields.size());
				continue;
			}
			tmp_fields.push_back(*it_curf);
			fields.erase(it_curf);
		}

		PVCore::list_fields &final_fields = f(tmp_fields);
		chunk_index ins_index = *(std::min_element(indx.begin(), indx.end()));
		PVCore::list_fields::iterator itins = fields.begin();
		std::advance(itins, ins_index);
		fields.splice(itins, final_fields);
	}
	return fields;
}
