/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVMAPPINGFILTERTIMEWEEK_H
#define PVFILTER_PVMAPPINGFILTERTIMEWEEK_H

#include <inendi/PVMappingFilter.h>

#include <pvkernel/rush/PVNraw.h>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <unicode/calendar.h>

#include <pvcop/types/datetime.h>

#include <omp.h>

namespace Inendi
{

class PVMappingFilterTimeWeek : public PVMappingFilter
{
	friend class time_mapping;

  public:
	PVMappingFilterTimeWeek();

  public:
	pvcop::db::array operator()(PVCol const col, PVRush::PVNraw const& nraw) override
	{
		auto f = nraw.collection().formatter(col);
		auto array = nraw.collection().column(col);
		pvcop::db::array dest(pvcop::db::type_uint32, array.size());
		auto& dest_array = dest.to_core_array<uint32_t>();

		if (std::string(f->name()) == "datetime") {
			auto& core_array = array.to_core_array<uint32_t>();
#pragma omp parallel for
			for (size_t row = 0; row < array.size(); row++) {
				tm local_tm;
				const time_t t = static_cast<int64_t>(core_array[row]);
				pvcop::types::formatter_datetime::gmtime_r(&t, &local_tm);

				dest_array[row] =
				    local_tm.tm_sec +
				    60 * (local_tm.tm_min + 60 * (local_tm.tm_hour + 24 * local_tm.tm_wday));
			}
		} else if (std::string(f->name()) == "datetime_us") {
			auto& core_array = array.to_core_array<uint64_t>();
#pragma omp parallel for
			for (size_t row = 0; row < array.size(); row++) {
				const boost::posix_time::ptime t =
				    *reinterpret_cast<const boost::posix_time::ptime*>(&core_array[row]);
				dest_array[row] = t.time_of_day().total_seconds() +
				                  60 * 60 * 24 * t.date().day_of_week().as_number();
			}
		} else {
			assert(std::string(f->name()) == "datetime_ms" && "Unknown datetime formatter");

			auto& core_array = array.to_core_array<uint64_t>();

			std::vector<std::unique_ptr<Calendar>> calendars;
			for (size_t i = 0; i < (size_t)omp_get_max_threads(); i++) {
				UErrorCode err = U_ZERO_ERROR;
				calendars.emplace_back(Calendar::createInstance(err));
				if (not U_SUCCESS(err)) {
					throw std::runtime_error("Can't create calendar to compute mapping");
				}
			}

#pragma omp parallel
			{
				std::unique_ptr<Calendar>& cal = calendars[omp_get_thread_num()];
				UErrorCode err = U_ZERO_ERROR;
#pragma omp for
				for (size_t row = 0; row < array.size(); row++) {
					cal->setTime(core_array[row], err);
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
					dest_array[row] = sec + 60 * (min + 60 * (hour + 24 * wday));
				}
			}
		}
		return dest;
	}

	std::unordered_set<std::string> list_usable_type() const override { return {"time"}; }

	QString get_human_name() const override { return QString("Week"); }

	pvcop::db::array get_minmax(pvcop::db::array const&, pvcop::db::selection const&) const override
	{
		pvcop::db::array res(pvcop::db::type_uint32, 2);
		auto res_array = res.to_core_array<uint32_t>();
		res_array[0] = 0;
		res_array[1] = 7 * 24 * 3600 - 1;
		return res;
	}

	CLASS_FILTER_NOPARAM(PVMappingFilterTimeWeek)
};
}

#endif
