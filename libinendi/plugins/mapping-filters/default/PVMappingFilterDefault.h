/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVMAPPINGFILTERDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERDEFAULT_H

#include <inendi/PVMappingFilter.h>

#include <boost/date_time/posix_time/posix_time.hpp>

namespace Inendi
{

class PVMappingFilterDefault;

/**
 * Signed integer mapping. It keeps integer values.
 */
class PVMappingFilterDefault : public PVMappingFilter
{
  public:
	PVMappingFilterDefault();

	/**
	 * Copy NRaw values (real integers value) as mapping value.
	 */
	pvcop::db::array operator()(PVCol const col, PVRush::PVNraw const& nraw) override
	{
		auto array = nraw.collection().column(col);

		auto f = nraw.collection().formatter(col);
		if (std::string(f->name()) == "datetime_us") {
			pvcop::db::array dest(pvcop::db::type_uint32, array.size());
			auto& dest_array = dest.to_core_array<uint32_t>();
			auto& core_array = array.to_core_array<uint64_t>();
			const boost::posix_time::ptime epoch = boost::posix_time::from_time_t(0);
			for (size_t row = 0; row < array.size(); row++) {
				const boost::posix_time::ptime t =
				    *reinterpret_cast<const boost::posix_time::ptime*>(&core_array[row]);
				dest_array[row] = (t - epoch).total_seconds();
			}
			return dest;
		} else if (std::string(f->name()) == "datetime_ms") {
			pvcop::db::array dest(pvcop::db::type_uint32, array.size());
			auto& dest_array = dest.to_core_array<uint32_t>();
			auto& core_array = array.to_core_array<uint64_t>();
			for (size_t row = 0; row < array.size(); row++) {
				// FIXME : We could keep more ms information for mapping values.
				// FIXME : We can keep original data if we accept uint64_t as mapping value.
				dest_array[row] = core_array[row] / 1000; // ms to s;
			}
			return dest;
		} else if (std::string(f->name()) == "string") {
			/**
			 * string arrays need an explicit copy because underlying type is
			 * 'string_index_t' but we want to expose it as 'uint32_t' for inspector
			 */
			pvcop::db::array dest(pvcop::db::type_uint32, array.size());
			auto& dest_array = dest.to_core_array<uint32_t>();
			auto& core_array = array.to_core_array<string_index_t>();
			std::copy(core_array.begin(), core_array.end(), dest_array.begin());
			return dest;
		}

		return array.copy();
	}

	std::unordered_set<std::string> list_usable_type() const override
	{
		return {"ipv4", "ipv6", "mac_address", "time", "number_float", "number_int32", "number_uint32", "string"};
	}

	/**
	 * Metainformation for this plugin.
	 */
	QString get_human_name() const override { return QString("Default"); }

	CLASS_FILTER_NOPARAM(PVMappingFilterDefault)
};
}

#endif
