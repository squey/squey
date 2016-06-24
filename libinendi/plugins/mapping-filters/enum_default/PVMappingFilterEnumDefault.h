/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef PVFILTER_PVMAPPINGFILTERENUMDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERENUMDEFAULT_H

#include <inendi/PVMappingFilter.h>

#include <pvcop/db/read_dict.h>

namespace Inendi
{

/**
 * Mapping class for enum type.
 *
 * This mapping is an equireparteed one.
 */
class PVMappingFilterEnumDefault : public PVMappingFilter
{
  public:
	PVMappingFilterEnumDefault();

	/**
	 * Compute distinct value and associate for each of the an equi-reparteed
	 * value between 0 and uint32_t MAX.
	 */
	pvcop::db::array operator()(PVCol const col, PVRush::PVNraw const& nraw) override
	{
		auto array = nraw.collection().column(col);
		auto& core_array = array.to_core_array<uint32_t>();

		pvcop::db::array dest(pvcop::db::type_uint32, array.size());

		// Apply this factor to make sure we use the full uint32 range.
		double extend_factor =
		    std::numeric_limits<uint32_t>::max() / (double)nraw.collection().dict(col)->size();

		auto& dest_array = dest.to_core_array<uint32_t>();
		for (size_t row = 0; row < array.size(); row++) {
			dest_array[row] = extend_factor * core_array[row];
		}

		return dest;
	}

	/**
	 * MetaInformations about this plugins.
	 */
	QString get_human_name() const override { return QString("Default"); }

	CLASS_FILTER_NOPARAM(PVMappingFilterEnumDefault)
};
}

#endif
