/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVMAPPINGFILTERTIMEDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERTIMEDEFAULT_H

#include <pvkernel/core/general.h>
#include <inendi/PVMappingFilter.h>

#include <boost/date_time/posix_time/posix_time.hpp>

namespace Inendi {

class PVMappingFilterTimeDefault: public PVMappingFilter
{
	friend class time_mapping;
public:
	PVMappingFilterTimeDefault();

	public:
		decimal_storage_type* operator()(PVCol const col, PVRush::PVNraw const& nraw) override {
			auto f = nraw.collection().formatter(col);
			auto array = nraw.collection().column(col);

			if (std::string(f->name()) == "datetime") {
				auto& core_array = array.to_core_array<uint32_t>();
				for(size_t row=0; row< array.size(); row++) {
					_dest[row].storage_as_uint() = core_array[row];
				}
			} else if (std::string(f->name()) == "datetime_us") {
				auto& core_array = array.to_core_array<uint64_t>();
				const boost::posix_time::ptime epoch = boost::posix_time::from_time_t(0);
				for(size_t row=0; row< array.size(); row++) {
					const boost::posix_time::ptime t = *reinterpret_cast<const boost::posix_time::ptime*>(&core_array[row]);
					_dest[row].storage_as_uint() = (t - epoch).total_seconds();
				}
			} else {
				assert(std::string(f->name()) == "datetime_ms" && "Unknown datetime formatter");
				auto& core_array = array.to_core_array<uint64_t>();
				for(size_t row=0; row< array.size(); row++) {
					_dest[row].storage_as_uint() = core_array[row] / 1000; // ms to s
				}
			}

			return _dest;
		}

		QString get_human_name() const override { return QString("Default"); }
		PVCore::DecimalType get_decimal_type() const override { return PVCore::IntegerType; }

	CLASS_FILTER_NOPARAM(PVMappingFilterTimeDefault)
};

}

#endif
