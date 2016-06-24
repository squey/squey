/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef PVFILTER_PVMAPPINGFILTERFLOAT_H
#define PVFILTER_PVMAPPINGFILTERFLOAT_H

#include <inendi/PVMappingFilter.h>

namespace Inendi
{

/**
 * Class to compute default mapping for float type.
 */
class PVMappingFilterFloatDefault : public PVMappingFilter
{
  public:
	/**
	 * Compute mapping value which is the same as float values.
	 *
	 * @warning : storage type have to be float.
	 */
	pvcop::db::array operator()(PVCol const col, PVRush::PVNraw const& nraw) override
	{
		auto array = nraw.collection().column(col);
		auto& core_array = array.to_core_array<float>();

		pvcop::db::array dest(pvcop::db::type_float, array.size());
		auto& dest_array = dest.to_core_array<float>();

		for (size_t row = 0; row < array.size(); row++) {
			dest_array[row] = core_array[row];
		}

		return dest;
	}

	/**
	 * MetaInformation of this plugins.
	 */
	QString get_human_name() const { return QString("Default"); }

	CLASS_FILTER_NOPARAM(PVMappingFilterFloatDefault)
};
}

#endif
