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
#include <type_traits>
#include <unordered_set>

namespace Inendi
{

template <typename T>
struct PVRangeSubSamplerIntervalType {
	using value_type = double;
};

template <>
struct PVRangeSubSamplerIntervalType<boost::posix_time::ptime> {
	using value_type = boost::posix_time::time_duration;
};

class PVRangeSubSampler
{
  private:
	struct SamplingParams {
		size_t first = 0;
		size_t last = 0;
		pvcop::db::array minmax = {};
		size_t min = 0;
		size_t max = 0;

		SamplingParams(size_t first = 0,
		               size_t last = 0,
		               const pvcop::db::array& minmax = {},
		               size_t min = 0,
		               size_t max = 0)
		    : first(first), last(last), minmax(minmax.copy()), min(min), max(max)
		{
		}

		bool operator==(const SamplingParams& rhs) const
		{
			return rhs.first == first and rhs.last == last and rhs.minmax == minmax and
			       rhs.min == min and rhs.max == max;
		}
		bool operator!=(const SamplingParams& rhs) const { return not(*this == rhs); }
	};

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
	                  const pvcop::db::selection& sel,
	                  size_t sampling_count = 2048);

	void set_sampling_count(size_t sampling_count);
	void set_selected_timeseries(const std::unordered_set<size_t>& selected_timeseries = {});

	size_t samples_count() const { return _sampling_count; }
	size_t total_count() const { return _time.size(); }
	size_t timeseries_count() const { return _timeseries.size(); }
	const std::vector<display_type>& averaged_timeserie(size_t index) const
	{
		return _avg_matrix[index];
	}
	const pvcop::db::array& minmax_time() const { return _minmax; }
	pvcop::db::array minmax_subrange(double first_ratio, double last_ratio);

	void
	subsample(double first_ratio, double last_ratio, double min_ratio = 0, double max_ratio = 0);
	void subsample(const pvcop::db::array& minmax, size_t min = 0, size_t max = 0);
	void resubsample(const std::unordered_set<size_t>& timeseries = {});
	bool valid() const;

  private:
	void allocate_internal_structures();
	void subsample(
	    size_t first, size_t last, const pvcop::db::array& minmax, size_t min = 0, size_t max = 0);

	template <typename T>
	void compute_ranges_values_count(size_t first, size_t last, const pvcop::db::array& minmax)
	{
		assert(_sampled_time.size() == _sampling_count - 1);
		assert(_ranges_values_counts.size() == _sampling_count);

		BENCH_START(compute_ranges_values_count);

		pvcop::core::array<T> core_sampled_time = _sampled_time.to_core_array<T>();
		pvcop::core::array<T> core_time = _time.to_core_array<T>();
		pvcop::core::array<T> core_minmax = minmax.to_core_array<T>();

		T min = core_minmax[0];
		T max = core_minmax[1];

		using interval_t = typename PVRangeSubSamplerIntervalType<T>::value_type;
		const interval_t& interval = (interval_t)(max - min) / (_sampling_count);
		for (size_t i = 0; i < _sampled_time.size(); i++) {
			core_sampled_time[i] = (T)(min + (interval * (i + 1)));
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

  public:
	sigc::signal<void()> _subsampled;

  private:
	size_t _sampling_count;

	const pvcop::db::array& _time;
	const std::vector<pvcop::core::array<value_type>> _timeseries;
	std::unordered_set<size_t> _selected_timeseries;
	std::vector<size_t> _timeseries_to_subsample;
	const pvcop::db::selection& _sel;

	pvcop::db::indexes _sorted_indexes;
	pvcop::core::array<uint32_t> _sort;

	pvcop::db::array _sampled_time;

	pvcop::db::array _minmax;

	std::vector<size_t> _ranges_values_counts;
	std::vector<std::vector<display_type>> _avg_matrix;

	SamplingParams _last_params;

	bool _reset = false;
};

} // namespace Inendi

#endif // __PVRANGESUBSAMPLER_H__
