/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVMAPPINGFILTERINTEGER_H
#define PVFILTER_PVMAPPINGFILTERINTEGER_H

#include <inendi/PVMappingFilter.h>

namespace Inendi
{

class PVMappingFilterIntegerDefault;

/**
 * Signed integer mapping. It keeps integer values.
 */
class PVMappingFilterIntegerDefault : public PVMappingFilter
{
  public:
	PVMappingFilterIntegerDefault();

	/**
	 * Copy NRaw values (real integers value) as mapping value.
	 */
	pvcop::db::array operator()(PVCol const col, PVRush::PVNraw const& nraw) override
	{
		auto array = nraw.collection().column(col);
		auto& core_array = array.to_core_array<int32_t>();
		pvcop::db::array dest(pvcop::db::type_int32, array.size());
		dest.to_core_array<int32_t>().copy_from(core_array, 0, array.size());

		return dest;
	}

	/**
	 * Metainformation for this plugin.
	 */
	QString get_human_name() const override { return QString("Signed decimal"); }

	CLASS_FILTER_NOPARAM(PVMappingFilterIntegerDefault)
};
}

#endif
