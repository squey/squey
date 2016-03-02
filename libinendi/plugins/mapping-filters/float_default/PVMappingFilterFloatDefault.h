/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef PVFILTER_PVMAPPINGFILTERFLOAT_H
#define PVFILTER_PVMAPPINGFILTERFLOAT_H

#include <inendi/PVMappingFilter.h>

namespace Inendi {

/**
 * Class to compute default mapping for float type.
 */
class PVMappingFilterFloatDefault: public PVMappingFilter
{
	public:
		/**
		 * Compute mapping value which is the same as float values.
		 *
		 * @warning : storage type have to be float.
		 */
		decimal_storage_type* operator()(PVCol const col, PVRush::PVNraw const& nraw) override {
			auto array = nraw.collection().column(col);

			for(size_t row=0; row< array.size(); row++) {
				Inendi::PVMappingFilter::decimal_storage_type ds;
				ds.storage_as_float() = array.to_core_array<float>()[row];
				_dest[row] = ds;
			}

			return _dest;
		}

		/**
		 * MetaInformation of this plugins.
		 */
		QString get_human_name() const { return QString("Default"); }
		PVCore::DecimalType get_decimal_type() const override { return PVCore::FloatType; }

	CLASS_FILTER_NOPARAM(PVMappingFilterFloatDefault)
};

}

#endif
