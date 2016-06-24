/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVMAPPINGFILTERTIMEDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERTIMEDEFAULT_H

#include <inendi/PVMappingFilter.h>

#include <boost/date_time/posix_time/posix_time.hpp>

namespace Inendi
{

class PVMappingFilterTimeDefault : public PVMappingFilter
{
	friend class time_mapping;

  public:
	PVMappingFilterTimeDefault();

  public:
	pvcop::db::array operator()(PVCol const col, PVRush::PVNraw const& nraw) override
	{
		auto f = nraw.collection().formatter(col);
		auto array = nraw.collection().column(col);
		pvcop::db::array dest(pvcop::db::type_uint32, array.size());
		auto& dest_array = dest.to_core_array<uint32_t>();

		if (std::string(f->name()) == "datetime") {
			auto& core_array = array.to_core_array<uint32_t>();
			for (size_t row = 0; row < array.size(); row++) {
				dest_array[row] = core_array[row];
			}
		} else if (std::string(f->name()) == "datetime_us") {
			auto& core_array = array.to_core_array<uint64_t>();
			const boost::posix_time::ptime epoch = boost::posix_time::from_time_t(0);
			for (size_t row = 0; row < array.size(); row++) {
				const boost::posix_time::ptime t =
				    *reinterpret_cast<const boost::posix_time::ptime*>(&core_array[row]);
				dest_array[row] = (t - epoch).total_seconds();
			}
		} else {
			assert(std::string(f->name()) == "datetime_ms" && "Unknown datetime formatter");
			auto& core_array = array.to_core_array<uint64_t>();
			for (size_t row = 0; row < array.size(); row++) {
				dest_array[row] = core_array[row] / 1000; // ms to s
			}
		}
		return dest;
	}

	QString get_human_name() const override { return QString("Default"); }

	CLASS_FILTER_NOPARAM(PVMappingFilterTimeDefault)
};
}

#endif
