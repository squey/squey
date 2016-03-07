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
#include <unicode/calendar.h>

namespace Inendi {

class PVMappingFilterTime24h: public PVMappingFilter
{
	friend class time_mapping;
public:
	PVMappingFilterTime24h();

	public:
		decimal_storage_type* operator()(PVCol const col, PVRush::PVNraw const& nraw) override {
			auto f = nraw.collection().formatter(col);
			auto array = nraw.collection().column(col);

			if (std::string(f->name()) == "datetime") {
				for(size_t row=0; row< array.size(); row++) {

					thread_local tm local_tm;
					const time_t t = static_cast<int64_t>(array.to_core_array<uint32_t>()[row]);
					gmtime_r(&t, &local_tm);

					Inendi::PVMappingFilter::decimal_storage_type ds;
					ds.storage_as_uint() = local_tm.tm_sec + local_tm.tm_min * 60 + local_tm.tm_hour * 60 * 60;
					_dest[row] = ds;
				}
			} else if (std::string(f->name()) == "datetime_us") {
				for(size_t row=0; row< array.size(); row++) {
					Inendi::PVMappingFilter::decimal_storage_type ds;
					const boost::posix_time::ptime t = *reinterpret_cast<const boost::posix_time::ptime*>(&array.to_core_array<uint64_t>()[row]);
					ds.storage_as_uint() = t.time_of_day().total_seconds();
					_dest[row] = ds;
				}
			} else {
				assert(std::string(f->name()) == "datetime_ms" && "Unknown datetime formatter");
				UErrorCode err = U_ZERO_ERROR;
				std::unique_ptr<Calendar> cal(Calendar::createInstance(err));
				for(size_t row=0; row< array.size(); row++) {
					Inendi::PVMappingFilter::decimal_storage_type ds;
					if (not U_SUCCESS(err)) {
						return nullptr;//false;
					}
					cal->setTime(array.to_core_array<uint64_t>()[row], err);
					if (not U_SUCCESS(err)) {
						return nullptr;//false;
					}
					//int32_t msec = cal->get(UCAL_MILLISECOND, err);
					int32_t sec = cal->get(UCAL_SECOND, err);
					int32_t min = cal->get(UCAL_MINUTE, err);
					int32_t hour = cal->get(UCAL_HOUR_OF_DAY, err);
					ds.storage_as_uint() = (sec +  (min * 60) + (hour * 60 * 60));
					//cal->computeMillisInDay();
					_dest[row] = ds;
				}
			}

			return _dest;
		}

		QString get_human_name() const override { return QString("24h"); }
		PVCore::DecimalType get_decimal_type() const override { return PVCore::IntegerType; }

	CLASS_FILTER_NOPARAM(PVMappingFilterTime24h)
};

}

#endif
