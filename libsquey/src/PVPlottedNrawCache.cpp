//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <squey/PVPlottedNrawCache.h>

#include <pvkernel/rush/PVNraw.h>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_sort.h>
#include <tbb/parallel_for.h>

#include <algorithm>

/*****************************************************************************
 * Squey::PVPlottedNrawCache::PVPlottedNrawCache
 *****************************************************************************/

Squey::PVPlottedNrawCache::PVPlottedNrawCache(const PVView& view,
                                               const PVCol col,
                                               const size_t size)
    : _cache(size), _view(view), _nraw(view.get_rushnraw_parent()), _col(col)
{
}

/*****************************************************************************
 * Squey::PVPlottedNrawCache::initialize
 *****************************************************************************/

void Squey::PVPlottedNrawCache::initialize()
{
	if (_entries.size() != 0) {
		return;
	}

	_cache.invalidate();

	using range_t = tbb::blocked_range<size_t>;

	const PVPlotted::uint_plotted_t plotted = _view.get_parent<PVPlotted>().get_plotted(_col);

	tbb::concurrent_unordered_map<value_type, PVRow> dict;

	/* parallel unique key computation
	 */
	tbb::parallel_for(range_t(0, plotted.size()), [&](const range_t& r) {
		for (size_t i = r.begin(); i < r.end(); ++i) {
			/**
			 * only the firstly inserted entry sets its value
			 */
			dict.insert({plotted[i], i});
		}
	});

	/* sequential insertion
	 */
	_entries.reserve(dict.size());
	for (const auto& e : dict) {
		_entries.emplace_back(e.first, e.second);
	}

	/* parallel ordering according to the key
	 */
	tbb::parallel_sort(_entries.begin(), _entries.end());
}

/*****************************************************************************
 * Squey::PVPlottedNrawCache::invalidate
 *****************************************************************************/

void Squey::PVPlottedNrawCache::invalidate()
{
	_cache.invalidate();
}

/*****************************************************************************
 * Squey::PVPlottedNrawCache::get
 *****************************************************************************/

const QString Squey::PVPlottedNrawCache::get(const int64_t v)
{
	if (_entries.size() == 0) {
		return {};
	}

	if (_cache.exist(v)) {
		return _cache.get(v);
	}

	PVRow r_global;

	/* checking out-of-range before anything
	 */
	if (v <= _entries.begin()->plotted) {
		r_global = _entries.begin()->row;
	} else if (v >= _entries.rbegin()->plotted) {
		r_global = _entries.rbegin()->row;
	} else {
		/* getting an iterator on the *not-less* than entry, i.e the one which is greater or equal.
		 */
		const auto it = std::lower_bound(_entries.begin(), _entries.end(), entry_t(v, 0));

		if (it->plotted == v) {
			r_global = it->row;
		} else {
			/* must now check for the nearest between *it and *(it-1)
			 */
			const auto itp = std::prev(it);

			if (llabs(v - it->plotted) < llabs(v - itp->plotted)) {
				r_global = it->row;
			} else {
				r_global = itp->row;
			}
		}
	}

	QString result = QString::fromStdString(_nraw.at_string(r_global, _col));

	_cache.insert(v, result);

	return result;
}
