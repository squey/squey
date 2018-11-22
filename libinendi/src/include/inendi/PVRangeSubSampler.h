/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2018
 */

#ifndef __PVRANGESUBSAMPLER_H__
#define __PVRANGESUBSAMPLER_H__

#include <pvcop/db/array.h>
#include <pvcop/types/datetime_us.h>
#include <pvkernel/core/inendi_bench.h> // for BENCH_END, BENCH_START
#include <inendi/PVPlotted.h>

#include <numeric>
#include <math.h>

namespace Inendi
{

class PVRangeSubSampler
{
  private:
	static constexpr const size_t reserved_bits = 2;
	using value_type = Inendi::PVPlotted::value_type;

  public:
	using display_type = uint16_t;
	static constexpr const size_t display_type_max_val =
	    (1 << (size_t)(std::numeric_limits<value_type>::digits -
	                   std::numeric_limits<display_type>::digits - reserved_bits)) -
	    1;

  public:
	static constexpr const display_type no_value =
	    0b01 << (std::numeric_limits<display_type>::digits - reserved_bits);
	static constexpr const display_type underflow_value =
	    0b10 << (std::numeric_limits<display_type>::digits - reserved_bits);
	static constexpr const display_type overflow_value =
	    0b11 << (std::numeric_limits<display_type>::digits - reserved_bits);

  public:
	PVRangeSubSampler(const pvcop::db::array& time,
	                  const std::vector<pvcop::core::array<value_type>>& timeseries,
	                  size_t sampling_count = 2048);

	void set_sampling_count(size_t sampling_count);

	size_t samples_count() const { return _sampling_count; }
	size_t total_count() const { return _time.size(); }
	size_t timeseries_count() const { return _timeseries.size(); }
	const std::vector<display_type>& averaged_timeserie(size_t index) { return _avg_matrix[index]; }
	const std::vector<display_type>& averaged_time() { return _time_iota; }
	const pvcop::db::array& minmax_time() const { return _minmax; }

	void
	subsample(double first_ratio, double last_ratio, double min_ratio = 0, double max_ratio = 0);
	void subsample(const pvcop::db::array& minmax, size_t min = 0, size_t max = 0);
	void subsample(size_t first = 0, size_t last = 0, size_t min = 0, size_t max = 0);

	void resubsample();

  private:
	void allocate_internal_structures();

	template <typename T>
	void compute_ranges_values_count(size_t first, size_t last)
	{
		BENCH_START(compute_ranges_values_count);

		pvcop::core::array<T> core_sampled_time = _sampled_time.to_core_array<T>();
		pvcop::core::array<T> core_time = _time.to_core_array<T>();

		T start;
		T end;
		if (_sort) {
			start = core_time[_sort[first]];
			end = core_time[_sort[last - 1]];
		} else {
			start = core_time[first];
			end = core_time[last - 1];
		}

		const auto& interval = (end - start) / (_sampling_count);
		for (size_t i = 0; i < _sampled_time.size(); i++) {
			core_sampled_time[i] = start + (interval * (i + 1));
		}

		size_t j = first;
		size_t old_j = j;
		for (size_t i = 0; i < core_sampled_time.size(); i++) {
			if (_sort) {
				while (core_time[_sort[j]] < core_sampled_time[i])
					j++;
			} else {
				while (core_time[j] < core_sampled_time[i])
					j++;
			}
			_ranges_values_counts[i] = j - old_j;
			old_j = j;
		}
		_ranges_values_counts[_ranges_values_counts.size() - 1] = last - old_j;

		BENCH_END(compute_ranges_values_count, "compute_ranges_values_count", _sampled_time.size(),
		          1, _sampled_time.size(), 1);
	}

	void compute_ranges_average(size_t first, size_t /*last*/, size_t min, size_t max);

  private:
	size_t _sampling_count;

	size_t _first = 0;
	size_t _last = 0;
	size_t _min = 0;
	size_t _max = 0;

	const pvcop::db::array& _time;
	const std::vector<pvcop::core::array<value_type>> _timeseries;

	pvcop::db::indexes _sorted_indexes;
	pvcop::core::array<uint32_t> _sort;

	pvcop::db::array _sampled_time;

	pvcop::db::array _minmax;

	std::vector<display_type> _time_iota; // FIXME

	std::vector<size_t> _ranges_values_counts;
	std::vector<std::vector<display_type>> _avg_matrix;
};

} // namespace Inendi

#endif // __PVRANGESUBSAMPLER_H__
