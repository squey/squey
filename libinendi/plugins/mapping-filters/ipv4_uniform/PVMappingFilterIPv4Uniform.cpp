/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVMappingFilterIPv4Uniform.h"

#include <pvkernel/core/PVUnicodeString.h>
#include <pvkernel/core/PVUnicodeString16.h>
#include <pvkernel/core/network.h>

#include <climits>

/*****************************************************************************
 * Inendi::PVMappingFilterIPv4Uniform::PVMappingFilterIPv4Uniform
 *****************************************************************************/

Inendi::PVMappingFilterIPv4Uniform::PVMappingFilterIPv4Uniform(PVCore::PVArgumentList const& args):
	PVMappingFilter()
{
	INIT_FILTER(PVMappingFilterIPv4Uniform, args);
}

/*****************************************************************************
 * DEFAULT_ARGS_FILTER(Inendi::PVMappingFilterIPv4Uniform)
 *****************************************************************************/

DEFAULT_ARGS_FILTER(Inendi::PVMappingFilterIPv4Uniform)
{
	PVCore::PVArgumentList args;

	return args;
}

/*****************************************************************************
 * void Inendi::PVMappingFilterIPv4Uniform::init
 *****************************************************************************/

void Inendi::PVMappingFilterIPv4Uniform::init()
{
}

/*****************************************************************************
 * void Inendi::PVMappingFilterIPv4Uniform::get_human_name
 *****************************************************************************/

QString Inendi::PVMappingFilterIPv4Uniform::get_human_name() const
{
	return QString("Uniform");
}

/*****************************************************************************
 * void Inendi::PVMappingFilterIPv4Uniform::get_decimal_type
 *****************************************************************************/

PVCore::DecimalType Inendi::PVMappingFilterIPv4Uniform::get_decimal_type() const
{
	return PVCore::UnsignedIntegerType;
}

/*****************************************************************************
 * Inendi::PVMappingFilterIPv4Uniform::operator()
 *****************************************************************************/

Inendi::PVMappingFilter::decimal_storage_type*
Inendi::PVMappingFilterIPv4Uniform::operator()(PVCol const c,
                                               PVRush::PVNraw const& nraw)
{
	/* first travers the nraw to save into _dest the IPv4 as uint32
	 */
	auto const& array = nraw.collection().column(c);
	auto const& core_array = array.to_core_array<uint32_t>();

	std::map<uint32_t, uint32_t> values;

#pragma omp parallel for
	for(size_t i=0; i<array.size(); i++) {
		_dest[i].storage_as_uint() = core_array[i];
		values[_dest[i].storage_as_uint()] = 0;
	}

	/* as std::map is ordered by its keys, we just have to iterate to
	 * compute the mapping values
	 */
	uint32_t value_count = values.size();

	uint32_t i = 0;
	for (auto& kv : values) {
		// Compute ration between 0 and 1 and match it to 0 -> max
		kv.second = std::numeric_limits<uint32_t>::max() * ((double)i / value_count);
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
 * IMPL_FILTER(Inendi::PVMappingFilterIPv4Uniform)
 *****************************************************************************/

IMPL_FILTER(Inendi::PVMappingFilterIPv4Uniform)
