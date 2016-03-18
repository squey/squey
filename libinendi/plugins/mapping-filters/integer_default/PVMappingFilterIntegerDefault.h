/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVMAPPINGFILTERINTEGER_H
#define PVFILTER_PVMAPPINGFILTERINTEGER_H

#include <pvkernel/core/general.h>
#include <inendi/PVMappingFilter.h>

namespace Inendi {

class PVMappingFilterIntegerDefault;

/**
 * Signed integer mapping. It keeps integer values.
 */
class PVMappingFilterIntegerDefault: public PVMappingFilter
{
	public:
		PVMappingFilterIntegerDefault();

		/**
		 * Copy NRaw values (real integers value) as mapping value.
		 */
		decimal_storage_type* operator()(PVCol const col, PVRush::PVNraw const& nraw) override {
			auto array = nraw.collection().column(col);
			auto& core_array = array.to_core_array<int32_t>();

			for(size_t row=0; row< array.size(); row++) {
				_dest[row].storage_as_int() = core_array[row];
			}

			return _dest;
		}

		/**
		 * Metainformation for this plugin.
		 */
		QString get_human_name() const override { return QString("Signed decimal"); }
		PVCore::DecimalType get_decimal_type() const override { return PVCore::IntegerType; }

	CLASS_FILTER_NOPARAM(PVMappingFilterIntegerDefault)
};

}

#endif
