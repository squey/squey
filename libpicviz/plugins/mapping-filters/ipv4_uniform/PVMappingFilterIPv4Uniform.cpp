/**
 * \file PVMappingFilterIPv4Uniform.cpp
 *
 * Copyright (C) Picviz Labs 2014
 */

#include "PVMappingFilterIPv4Uniform.h"

#include <pvkernel/core/PVUnicodeString.h>
#include <pvkernel/core/PVUnicodeString16.h>
#include <pvkernel/core/network.h>

#include <climits>

/*****************************************************************************
 * Picviz::PVMappingFilterIPv4Uniform::PVMappingFilterIPv4Uniform
 *****************************************************************************/

Picviz::PVMappingFilterIPv4Uniform::PVMappingFilterIPv4Uniform(PVCore::PVArgumentList const& args):
	PVMappingFilter()
{
	INIT_FILTER(PVMappingFilterIPv4Uniform, args);
}

/*****************************************************************************
 * DEFAULT_ARGS_FILTER(Picviz::PVMappingFilterIPv4Uniform)
 *****************************************************************************/

DEFAULT_ARGS_FILTER(Picviz::PVMappingFilterIPv4Uniform)
{
	PVCore::PVArgumentList args;

	return args;
}

/*****************************************************************************
 * void Picviz::PVMappingFilterIPv4Uniform::init
 *****************************************************************************/

void Picviz::PVMappingFilterIPv4Uniform::init()
{
	_values.clear();
}

/*****************************************************************************
 * void Picviz::PVMappingFilterIPv4Uniform::get_human_name
 *****************************************************************************/

QString Picviz::PVMappingFilterIPv4Uniform::get_human_name() const
{
	return QString("Uniform");
}

/*****************************************************************************
 * void Picviz::PVMappingFilterIPv4Uniform::get_decimal_type
 *****************************************************************************/

PVCore::DecimalType Picviz::PVMappingFilterIPv4Uniform::get_decimal_type() const
{
	return PVCore::UnsignedIntegerType;
}

/*****************************************************************************
 * Picviz::PVMappingFilterIPv4Uniform::operator()
 *****************************************************************************/

Picviz::PVMappingFilter::decimal_storage_type*
Picviz::PVMappingFilterIPv4Uniform::operator()(PVCol const c,
                                               PVRush::PVNraw const& nraw)
{
	/* first travers once to have distinct values
	 */
	nraw.visit_column(c, [&](PVRow, const char* buf, size_t size)
	                  {
		                  uint32_t ipv4;
		                  PVCore::Network::ipv4_aton(buf, size, ipv4);

		                  typename values_t::iterator it = _values.find(ipv4);
		                  if (it == _values.end()) {
			                  _values[ipv4] = 0;
		                  }
	                  });

	/* then ::finalize() do the rest
	 */
	return finalize(c, nraw);
}

/*****************************************************************************
 * Picviz::PVMappingFilterIPv4Uniform::operator()
 *****************************************************************************/

Picviz::PVMappingFilter::decimal_storage_type
Picviz::PVMappingFilterIPv4Uniform::operator()(PVCore::PVField const& field)
{
	Picviz::PVMappingFilter::decimal_storage_type ret_ds;
	uint32_t ipv4;
	PVCore::Network::ipv4_a16ton((uint16_t*)field.begin(), field.size()/sizeof(uint16_t), ipv4);

	typename values_t::iterator it = _values.find(ipv4);
	if (it == _values.end()) {
		_values[ipv4] = 0;
	}

	ret_ds.storage_as_uint() = 0;

	return ret_ds;
}

/*****************************************************************************
 * Picviz::PVMappingFilterIPv4Uniform::finalize
 *****************************************************************************/

Picviz::PVMappingFilter::decimal_storage_type*
Picviz::PVMappingFilterIPv4Uniform::finalize(PVCol const c,
                                             PVRush::PVNraw const& nraw)
{
	/* _values contains all the distinct values
	 */

	/* first we sort the keys and set the mapped values
	 */
	auto key_list = _values.keys();
	qSort(key_list.begin(), key_list.end());

	uint32_t value_count = key_list.size();

	uint32_t i = 0;
	for (const auto& key : key_list) {
		_values[key] = (std::numeric_limits<uint32_t>::max() * (double)i) / value_count;
		++i;
	}

	/* then the mapping array is filled
	 */
	nraw.visit_column(c, [&](PVRow i, const char* buf, size_t size)
	                  {
		                  uint32_t ipv4;
		                  PVCore::Network::ipv4_aton(buf, size, ipv4);
		                  _dest[i].storage_as_uint() = _values.find(ipv4).value();
	                  });

	return _dest;
}

/*****************************************************************************
 * IMPL_FILTER(Picviz::PVMappingFilterIPv4Uniform)
 *****************************************************************************/

IMPL_FILTER(Picviz::PVMappingFilterIPv4Uniform)
