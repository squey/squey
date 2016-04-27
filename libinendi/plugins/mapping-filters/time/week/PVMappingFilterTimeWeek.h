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

namespace Inendi
{

class PVMappingFilterTimeWeek : public PVMappingFilter
{
	friend class time_mapping;

  public:
	PVMappingFilterTimeWeek();

  public:
	decimal_storage_type* operator()(PVCol const col, PVRush::PVNraw const& nraw) override
	{
		auto f = nraw.collection().formatter(col);
		auto array = nraw.collection().column(col);

		if (std::string(f->name()) == "datetime") {
			auto& core_array = array.to_core_array<uint32_t>();
			for (size_t row = 0; row < array.size(); row++) {
				tm local_tm;
				const time_t t = static_cast<int64_t>(core_array[row]);
				gmtime_r(&t, &local_tm);

				_dest[row].storage_as_uint() =
				    local_tm.tm_sec +
				    60 * (local_tm.tm_min + 60 * (local_tm.tm_hour + 24 * local_tm.tm_wday));
			}
		} else if (std::string(f->name()) == "datetime_us") {
			auto& core_array = array.to_core_array<uint64_t>();
			for (size_t row = 0; row < array.size(); row++) {
				const boost::posix_time::ptime t =
				    *reinterpret_cast<const boost::posix_time::ptime*>(&core_array[row]);
				_dest[row].storage_as_uint() = t.time_of_day().total_seconds() +
				                               60 * 60 * 24 * t.date().day_of_week().as_number();
			}
		} else {
			assert(std::string(f->name()) == "datetime_ms" && "Unknown datetime formatter");
			UErrorCode err = U_ZERO_ERROR;
			std::unique_ptr<Calendar> cal(Calendar::createInstance(err));
			if (not U_SUCCESS(err)) {
				throw std::runtime_error("Can't create calendar to compute mapping.");
			}

			for (size_t row = 0; row < array.size(); row++) {
				cal->setTime(array.to_core_array<uint64_t>()[row], err);
				if (not U_SUCCESS(err)) {
					continue;
				}
				int32_t sec = cal->get(UCAL_SECOND, err);
				if (not U_SUCCESS(err)) {
					continue;
				}
				int32_t min = cal->get(UCAL_MINUTE, err);
				if (not U_SUCCESS(err)) {
					continue;
				}
				int32_t hour = cal->get(UCAL_HOUR_OF_DAY, err);
				if (not U_SUCCESS(err)) {
					continue;
				}
				int32_t wday = cal->get(UCAL_DAY_OF_WEEK, err);
				if (not U_SUCCESS(err)) {
					continue;
				}
				_dest[row].storage_as_uint() = sec + 60 * (min + 60 * (hour + 24 * wday));
			}
		}

		return _dest;
	}

	QString get_human_name() const override { return QString("Week"); }
	PVCore::DecimalType get_decimal_type() const override { return PVCore::IntegerType; }

	CLASS_FILTER_NOPARAM(PVMappingFilterTimeWeek)
};
}

#endif
