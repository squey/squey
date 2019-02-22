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
#include <pvkernel/rush/PVNraw.h>

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
  public:
	using zoom_f = double;

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
	                  const PVRush::PVNraw& nraw,
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
	const std::vector<size_t>& histogram() const { return _histogram; }
	const pvcop::db::array& minmax_time() const { return _minmax; }
	pvcop::db::array minmax_subrange(zoom_f first_ratio, zoom_f last_ratio) const;
	std::pair<zoom_f, zoom_f> minmax_ratio(const pvcop::db::array& minmax) const;
	const pvcop::db::indexes& sorted_indexes() const { return _sorted_indexes; }

	void
	subsample(zoom_f first_ratio, zoom_f last_ratio, zoom_f min_ratio = 0, zoom_f max_ratio = 0);
	void subsample(const pvcop::db::array& minmax, uint32_t min = 0, uint32_t max = 0);
	void resubsample(const std::unordered_set<size_t>& timeseries = {});
	bool valid() const;

  private:
	void allocate_internal_structures();
	void subsample(size_t first,
	               size_t last,
	               const pvcop::db::array& minmax,
	               uint32_t min = 0,
	               uint32_t max = 0);

	template <typename T>
	void compute_histogram(size_t first, size_t last, const pvcop::db::array& minmax)
	{
		assert(_histogram.size() == _sampling_count);

		BENCH_START(compute_histogram);

		pvcop::core::array<T> core_time = _time.to_core_array<T>();
		pvcop::core::array<T> core_minmax = minmax.to_core_array<T>();

		T min = core_minmax[0];
		T max = core_minmax[1];

		const pvcop::db::selection& invalid_sel = _time.invalid_selection();

		using interval_t = typename PVRangeSubSamplerIntervalType<T>::value_type;
		const interval_t& interval = (interval_t)(max - min) / (_sampling_count);

		auto sorted_begin_it = _sort.cbegin() + first;
		auto sorted_end_it = _sort.cbegin() + last + 1;
		auto begin_it = core_time.cbegin() + first;
		auto end_it = core_time.cbegin() + last + 1;

		//#pragma omp parallel for firstprivate(sorted_begin_it, begin_it) // working but slower
		for (size_t i = 0; i < _sampling_count - 1; i++) {
			const T value = (T)(min + (interval * (i + 1)));
			if (_sort) {
				sorted_begin_it = std::lower_bound(
				    sorted_begin_it, sorted_end_it, 0,
				    [this, &core_time, &invalid_sel, value](size_t j, size_t) {
					    return core_time[j] < value and (not invalid_sel or not invalid_sel[j]);
					});
				begin_it = core_time.cbegin() + std::distance(_sort.cbegin(), sorted_begin_it);
			} else {
				begin_it = std::lower_bound(begin_it, end_it, value);
			}
			_histogram[i] = std::distance(core_time.cbegin(), begin_it);
		}
		_histogram.back() = last + 1;

		// Transform indexes into histogram
		size_t first_range = _histogram.front() - first;
		std::adjacent_difference(_histogram.begin(), _histogram.end(), _histogram.begin());
		_histogram.front() = first_range;

		BENCH_END(compute_histogram, "compute_histogram", _sampling_count, 1, _sampling_count, 1);
	}

	void compute_ranges_average(size_t first, size_t /*last*/, size_t min, size_t max);

  public:
	sigc::signal<void()> _subsampled;

  private:
	size_t _sampling_count;

	const pvcop::db::array& _time;
	const std::vector<pvcop::core::array<value_type>> _timeseries;
	const PVRush::PVNraw& _nraw;
	std::unordered_set<size_t> _selected_timeseries;
	std::vector<size_t> _timeseries_to_subsample;
	const pvcop::db::selection& _sel;

	pvcop::db::indexes _sorted_indexes;
	pvcop::core::array<uint32_t> _sort;

	pvcop::db::array _minmax;

	std::vector<size_t> _histogram;
	std::vector<std::vector<display_type>> _avg_matrix;

	SamplingParams _last_params;

	bool _reset = false;
};

} // namespace Inendi

#endif // __PVRANGESUBSAMPLER_H__
