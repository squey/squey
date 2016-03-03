/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVMAPPINGFILTERIPV4_DEFAULT_H
#define PVFILTER_PVMAPPINGFILTERIPV4_DEFAULT_H

#include <pvkernel/core/general.h>
#include <inendi/PVMappingFilter.h>

namespace Inendi {

class PVMappingFilterIPv4Default;

/**
 * IPv4 mapping. It keeps unsigned integer values.
 */
class PVMappingFilterIPv4Default: public PVMappingFilter
{
	public:
		PVMappingFilterIPv4Default();

		/**
		 * Copy NRaw values (real unsigned integers value) as mapping value.
		 */
		decimal_storage_type* operator()(PVCol const col, PVRush::PVNraw const& nraw) override {
			auto array = nraw.collection().column(col);

			for(size_t row=0; row< array.size(); row++) {
				Inendi::PVMappingFilter::decimal_storage_type ds;
				ds.storage_as_int() = array.to_core_array<uint32_t>()[row];
				_dest[row] = ds;
			}

			return _dest;
		}

		/**
		 * Metainformation for this plugin.
		 */
		QString get_human_name() const override { return QString("Default"); }
		PVCore::DecimalType get_decimal_type() const override { return PVCore::UnsignedIntegerType; }

	CLASS_FILTER_NOPARAM(PVMappingFilterIPv4Default)
};

}

#endif
