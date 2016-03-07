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
				for(size_t row=0; row< array.size(); row++) {
					Inendi::PVMappingFilter::decimal_storage_type ds;
					ds.storage_as_uint() = array.to_core_array<uint32_t>()[row];
					_dest[row] = ds;
				}
			} else if (std::string(f->name()) == "datetime_us") {
				for(size_t row=0; row< array.size(); row++) {
					Inendi::PVMappingFilter::decimal_storage_type ds;
					const boost::posix_time::ptime t = *reinterpret_cast<const boost::posix_time::ptime*>(&array.to_core_array<uint64_t>()[row]);
					ds.storage_as_uint() = (t - boost::posix_time::from_time_t(0)).total_seconds();
					_dest[row] = ds;
				}
			} else {
				assert(std::string(f->name()) == "datetime_ms" && "Unknown datetime formatter");
				for(size_t row=0; row< array.size(); row++) {
					Inendi::PVMappingFilter::decimal_storage_type ds;
					ds.storage_as_uint() = array.to_core_array<uint64_t>()[row] / 1000;
					_dest[row] = ds;
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
