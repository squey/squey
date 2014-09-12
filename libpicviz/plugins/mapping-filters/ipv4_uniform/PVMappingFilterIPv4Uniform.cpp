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

typedef std::map<uint32_t, uint32_t> values_t;

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
	/* first travers the nraw to save into _dest the IPv4 as uint32
	 */
	nraw.visit_column(c, [&](PVRow r, const char* buf, size_t size)
	                  {
		                  uint32_t ipv4;
		                  PVCore::Network::ipv4_aton(buf, size, ipv4);
		                  _dest[r].storage_as_uint() = ipv4;
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
	/* simply save IPv4 as uint32 in result's op
	 */
	Picviz::PVMappingFilter::decimal_storage_type ret_ds;
	uint32_t ipv4;

	PVCore::Network::ipv4_a16ton((uint16_t*)field.begin(), field.size()/sizeof(uint16_t), ipv4);
	ret_ds.storage_as_uint() = ipv4;

	return ret_ds;
}

/*****************************************************************************
 * Picviz::PVMappingFilterIPv4Uniform::finalize
 *****************************************************************************/

Picviz::PVMappingFilter::decimal_storage_type*
Picviz::PVMappingFilterIPv4Uniform::finalize(PVCol const,
                                             PVRush::PVNraw const&)
{
	values_t values;

	/* populate values with _dest (which contains IPv4 as uint32
	 */
	for(size_t i = 0; i < _dest_size; ++i) {
		values[_dest[i].storage_as_uint()] = 0;
	}

	/* as std::map is ordered by its keys, we just have to iterate to
	 * compute the mapping values
	 */
	uint32_t value_count = values.size();

	uint32_t i = 0;
	for (auto& kv : values) {
		kv.second = (std::numeric_limits<uint32_t>::max() * (double)i) / value_count;
		++i;
	}

	/* then the mapping array is iterated to replace the numeric IPv4
	 * with its mapping value
	 */
#pragma omp parallel for
	for(size_t i = 0; i < _dest_size; ++i) {
		uint32_t ipv4 = _dest[i].storage_as_uint();
		_dest[i].storage_as_uint() = values.find(ipv4)->second;
	}

	return _dest;
}

/*****************************************************************************
 * IMPL_FILTER(Picviz::PVMappingFilterIPv4Uniform)
 *****************************************************************************/

IMPL_FILTER(Picviz::PVMappingFilterIPv4Uniform)
