/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __PVRANGESUBSAMPLER_H__
#define __PVRANGESUBSAMPLER_H__

#include <pvcop/db/array.h>
#include <pvcop/types/datetime_us.h>
#include <pvkernel/core/squey_bench.h> // for BENCH_END, BENCH_START
#include <squey/PVScaled.h>
#include <pvkernel/rush/PVNraw.h>

#include <numeric>
#include <math.h>
#include <type_traits>
#include <unordered_set>

namespace Squey
{

class PVRangeSubSampler
{
  public:
	enum SAMPLING_MODE {
		MEAN,
		MIN,
		MAX,
	};

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
	using value_type = Squey::PVScaled::value_type;

  public:
	using display_type = uint16_t;
	static constexpr const size_t display_value_bits =
	    std::numeric_limits<display_type>::digits - reserved_bits;
	static constexpr const display_type display_type_min_val = 0;
	static constexpr const display_type display_type_max_val = (1 << display_value_bits) - 1;

  public:
	static constexpr const display_type no_value = 0b01 << display_value_bits;
	static constexpr const display_type underflow_value = 0b10 << display_value_bits;
	static constexpr const display_type overflow_value = 0b11 << display_value_bits;

	static constexpr bool display_match(display_type d, display_type mask)
	{
		return ((d >> display_value_bits) << display_value_bits) == mask;
	}

  public:
	PVRangeSubSampler(const pvcop::db::array& time,
	                  const std::vector<pvcop::core::array<value_type>>& timeseries,
	                  const PVRush::PVNraw& nraw,
	                  const pvcop::db::selection& sel,
	                  const pvcop::db::array* split = nullptr,
	                  size_t sampling_count = 2048);

	template <SAMPLING_MODE mode>
	void set_sampling_mode()
	{
		_compute_ranges_reduction_f = [&](auto... args) {
			compute_ranges_reduction<mode>(args...);
		};
	}
	void set_sampling_count(size_t sampling_count);
	void set_selected_timeseries(const std::unordered_set<size_t>& selected_timeseries);
	void set_split_column(const pvcop::db::array* split);

	size_t samples_count() const { return _sampling_count; }
	size_t total_count() const { return _time.get().size(); }
	size_t timeseries_count() const { return _timeseries.size(); }
	size_t group_count() const { return _split_count; }
	std::string group_name(size_t i) const
	{
		return _split ? _split->at(_split_extents.to_core_array()[i]) : "";
	}
	const std::vector<display_type>& sampled_timeserie(size_t index) const
	{
		return _ts_matrix[index];
	}
	const std::vector<size_t>& histogram() const { return _histogram; }
	const pvcop::db::array& minmax_time() const { return _minmax; }
	pvcop::db::array ratio_to_minmax(zoom_f ratio1, zoom_f ratio2) const;
	std::pair<zoom_f, zoom_f> minmax_to_ratio(const pvcop::db::array& minmax) const;
	const pvcop::db::indexes& sorted_indexes() const { return _sorted_indexes; }

	void
	subsample(zoom_f first_ratio, zoom_f last_ratio, zoom_f min_ratio = 0, zoom_f max_ratio = 0);
	void subsample(const pvcop::db::array& minmax, uint32_t min = 0, uint32_t max = 0);
	void resubsample();
	void resubsample(const std::unordered_set<size_t>& timeseries);
	bool valid() const;

  private:
	void allocate_internal_structures();
	void subsample(size_t first,
	               size_t last,
	               const pvcop::db::array& minmax,
	               uint32_t min = 0,
	               uint32_t max = 0);

	template <SAMPLING_MODE mode>
	void compute_ranges_reduction(size_t first, size_t /*last*/, size_t min, size_t max);

	template <typename F>
	void compute_ranges_reduction(size_t first, size_t /*last*/, size_t min, size_t max);

  public:
	sigc::signal<void()> _subsampled;

  private:
	size_t _sampling_count;

	const pvcop::db::array& _original_time;
	std::reference_wrapper<const pvcop::db::array> _time;
	const std::vector<pvcop::core::array<value_type>> _timeseries;
	const PVRush::PVNraw& _nraw;
	std::unordered_set<size_t> _selected_timeseries;
	std::vector<size_t> _timeseries_to_subsample;
	const pvcop::db::selection& _sel;
	const pvcop::db::array* _split;
	pvcop::db::groups _split_groups;
	pvcop::db::extents _split_extents;
	size_t _split_count = 1;

	pvcop::db::indexes _sorted_indexes;
	pvcop::core::array<uint32_t> _sort;

	pvcop::db::array _minmax;

	std::vector<size_t> _histogram;
	std::vector<std::vector<display_type>> _ts_matrix;

	SamplingParams _last_params;

	bool _reset = false;
	bool _valid = false;

	std::function<void(size_t, size_t, size_t, size_t)> _compute_ranges_reduction_f;
};

template <Squey::PVRangeSubSampler::SAMPLING_MODE M>
struct sampling_mode_t {
	static constexpr const Squey::PVRangeSubSampler::SAMPLING_MODE mode = M;
	inline static uint64_t init() { return 0; }
	inline static uint64_t reduce(uint64_t accum, uint32_t) { return accum; }
};

template <Squey::PVRangeSubSampler::SAMPLING_MODE mode, typename... T>
struct func_resolver {
	using type = std::tuple_element_t<mode, std::tuple<T...>>;
	static_assert(type::mode == mode,
	              "func_resolver parameters must be ordered according to SAMPLING_MODE enum");
};

template <Squey::PVRangeSubSampler::SAMPLING_MODE mode>
void Squey::PVRangeSubSampler::compute_ranges_reduction(size_t first,
                                                         size_t /*last*/,
                                                         size_t min,
                                                         size_t max)
{

	struct mean_t : sampling_mode_t<SAMPLING_MODE::MEAN> {
		inline static void map(uint64_t& accum, uint32_t value)
		{
			accum += (std::numeric_limits<uint32_t>::max() - value);
		}
		inline static uint64_t reduce(uint64_t accum, uint32_t value_count)
		{
			return accum / value_count;
		}
	};

	struct min_t : sampling_mode_t<SAMPLING_MODE::MIN> {
		inline static uint64_t init() { return std::numeric_limits<uint64_t>::max(); }
		inline static void map(uint64_t& accum, uint32_t value)
		{
			accum = std::min(std::numeric_limits<uint32_t>::max() - value, (uint32_t)accum);
		}
	};

	struct max_t : sampling_mode_t<SAMPLING_MODE::MAX> {
		inline static void map(uint64_t& accum, uint32_t value)
		{
			accum = std::max(std::numeric_limits<uint32_t>::max() - value, (uint32_t)accum);
		}
	};

	compute_ranges_reduction<typename func_resolver<mode, mean_t, min_t, max_t>::type>(
	    first, (size_t)0, min, max);
}

template <typename F>
void Squey::PVRangeSubSampler::compute_ranges_reduction(size_t first,
                                                         size_t /*last*/,
                                                         size_t min,
                                                         size_t max)
{
	BENCH_START(compute_ranges_reduction);

	// Remove invalid values from selection
	const pvcop::db::selection& valid_sel = _time.get().valid_selection(_sel);

	const auto split_groups =
	    _split ? _split_groups.to_core_array() : pvcop::core::array<pvcop::db::index_t>();
	std::vector<uint64_t> accums(_split_count, F::init());
	std::vector<uint64_t> selected_values_counts(_split_count, 0);
	std::unordered_set<size_t> columns_to_subsample_set;
	for (size_t t : _timeseries_to_subsample) {
		columns_to_subsample_set.emplace(t / _split_count);
	}
	const std::vector<size_t> columns_to_subsample(columns_to_subsample_set.begin(),
	                                               columns_to_subsample_set.end());

#pragma omp parallel for firstprivate(accums, selected_values_counts)
	for (auto it = columns_to_subsample.begin(); it < columns_to_subsample.end(); ++it) {
		size_t i = *it;
		size_t start = first;
		size_t end = first;
		const pvcop::core::array<value_type>& timeserie = _timeseries[i];
		const pvcop::db::selection& ts_valid_sel =
		    _nraw.column(PVCol(i)).valid_selection(valid_sel);
		for (size_t j = 0; j < _histogram.size(); j++) {
			const size_t values_count = _histogram[j];
			end += values_count;
			std::fill(selected_values_counts.begin(), selected_values_counts.end(), 0);
			std::fill(accums.begin(), accums.end(), F::init());
			for (size_t k = start; k < end; k++) {
				auto v = not _sort ? k : _sort[k];
				const size_t group_index = _split ? split_groups[v] : 0;
				uint64_t& accum = accums[group_index];
				if (ts_valid_sel[v]) {
					selected_values_counts[group_index]++;
					F::map(accum, timeserie[v]);
				}
			}
			start = end;
			for (size_t group_index = 0; group_index < _split_count; group_index++) {
				size_t& selected_values_count = selected_values_counts[group_index];
				const size_t ii = (_split_count * i) + group_index;
				if (selected_values_count == 0) {
					_ts_matrix[ii][j] = no_value; // no value in range
				} else {
					uint64_t& accum = accums[group_index];
					const uint64_t raw_value = F::reduce(accum, selected_values_count);
					if (min != 0 and raw_value < min) { // underflow
						_ts_matrix[ii][j] = underflow_value;
					} else if (raw_value > max) { // overflow
						_ts_matrix[ii][j] = overflow_value;
					} else {
						_ts_matrix[ii][j] = (display_type)((zoom_f(raw_value - min) / (max - min)) *
						                                   display_type_max_val); // nominal value
					}
				}
			}
		}
	}

	BENCH_END(compute_ranges_reduction, "compute_ranges_reduction", _time.get().size(),
	          sizeof(uint64_t), _sampling_count, sizeof(uint64_t));
}

} // namespace Squey

#endif // __PVRANGESUBSAMPLER_H__
