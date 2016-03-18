/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef PVFILTER_PVMAPPINGFILTERIPV4UNIFORM_H
#define PVFILTER_PVMAPPINGFILTERIPV4UNIFORM_H

#include <inendi/PVMappingFilter.h>

#include <QString>

namespace Inendi {

/**
 * Mapping class for uniform IP.v4
 *
 * This mapping is an equireparteed one.
 */
class PVMappingFilterIPv4Uniform: public PVMappingFilter
{
	public:
		PVMappingFilterIPv4Uniform();

		/**
		 * Compute distinct value and associate for each of the an equi-reparteed 
		 * value between 0 and uint32_t MAX.
		 */
		decimal_storage_type* operator()(PVCol const col, PVRush::PVNraw const& nraw) override {
			auto array = nraw.collection().column(col);
			pvcop::db::groups group;
			pvcop::db::extents extents;

			array.group(group, extents);

			auto& core_group = group.to_core_array();

			double extend_factor = std::numeric_limits<uint32_t>::max() / (double)extents.size();
			for(size_t row=0; row< array.size(); row++) {
				_dest[row].storage_as_uint() = extend_factor * core_group[row];
			}

			return _dest;
		}

		/**
		 * MetaInformations about this plugins.
		 */
		QString get_human_name() const override { return QString("Uniform"); }
		PVCore::DecimalType get_decimal_type() const override { return PVCore::UnsignedIntegerType; }

		CLASS_FILTER_NOPARAM(PVMappingFilterIPv4Uniform)
};

}

#endif
