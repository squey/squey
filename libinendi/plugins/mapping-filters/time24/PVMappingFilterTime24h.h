/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVMAPPINGFILTERTIME24H_H
#define PVFILTER_PVMAPPINGFILTERTIME24H_H

#include <inendi/PVMappingFilter.h>

#include <pvkernel/rush/PVNraw.h>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <unicode/calendar.h>

#include <pvcop/types/datetime.h>

#include <omp.h>

namespace Inendi
{

class PVMappingFilterTime24h : public PVMappingFilter
{
  public:
	PVMappingFilterTime24h();

  public:
	pvcop::db::array operator()(PVCol const col, PVRush::PVNraw const& nraw) override
	{
		auto f = nraw.column(col).formatter();
		const pvcop::db::array& array = nraw.column(col);
		pvcop::db::array dest("number_uint32", array.size());
		auto& dest_array = dest.to_core_array<uint32_t>();

		if (std::string(f->name()) == "datetime") {
			auto& core_array = array.to_core_array<uint32_t>();
#pragma omp parallel for
			for (size_t row = 0; row < array.size(); row++) {
				tm local_tm;
				const time_t t = static_cast<int64_t>(core_array[row]);
				pvcop::types::formatter_datetime::gmtime_r(&t, &local_tm);

				dest_array[row] =
				    (((local_tm.tm_hour * 60) + local_tm.tm_min) * 60 + local_tm.tm_sec) * 1000;
			}
		} else if (std::string(f->name()) == "datetime_us") {
			auto& core_array = array.to_core_array<uint64_t>();
#pragma omp parallel for
			for (size_t row = 0; row < array.size(); row++) {
				const boost::posix_time::ptime t =
				    *reinterpret_cast<const boost::posix_time::ptime*>(&core_array[row]);
				const auto& tod = t.time_of_day();
				dest_array[row] = (tod.total_seconds() * 1000) + (tod.fractional_seconds() / 1000);
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

					int32_t millisec = cal->get(UCAL_MILLISECOND, err);
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

					dest_array[row] = ((sec + (min * 60) + (hour * 60 * 60)) * 1000) + millisec;
				}
			}
		}
		return dest;
	}

	std::unordered_set<std::string> list_usable_type() const override { return {"time"}; }

	QString get_human_name() const override { return QString("24h"); }

	pvcop::db::array get_minmax(pvcop::db::array const&, pvcop::db::selection const&) const override
	{
		pvcop::db::array res("number_uint32", 2);
		auto res_array = res.to_core_array<uint32_t>();
		res_array[0] = 0;
		res_array[1] = (24 * 60 * 60 * 1000) - 1;
		return res;
	}

	CLASS_FILTER_NOPARAM(PVMappingFilterTime24h)
};
}

#endif
