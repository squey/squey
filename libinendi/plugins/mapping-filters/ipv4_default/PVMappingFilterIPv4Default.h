/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVMAPPINGFILTERIPV4_DEFAULT_H
#define PVFILTER_PVMAPPINGFILTERIPV4_DEFAULT_H

#include <inendi/PVMappingFilter.h>

namespace Inendi
{

class PVMappingFilterIPv4Default;

/**
 * IPv4 mapping. It keeps unsigned integer values.
 */
class PVMappingFilterIPv4Default : public PVMappingFilter
{
  public:
	PVMappingFilterIPv4Default();

	/**
	 * Copy NRaw values (real unsigned integers value) as mapping value.
	 */
	pvcop::db::array operator()(PVCol const col, PVRush::PVNraw const& nraw) override
	{
		auto array = nraw.collection().column(col);
		auto& core_array = array.to_core_array<uint32_t>();
		pvcop::db::array dest(pvcop::db::type_uint32, array.size());
		dest.to_core_array<uint32_t>().copy_from(core_array, 0, array.size());

		return dest;
	}

	/**
	 * Metainformation for this plugin.
	 */
	QString get_human_name() const override { return QString("Default"); }

	CLASS_FILTER_NOPARAM(PVMappingFilterIPv4Default)
};
}

#endif
