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

#ifndef SQUEY_PVPLOTTEDNRAWCACHE_H
#define SQUEY_PVPLOTTEDNRAWCACHE_H

#include <pvbase/types.h>

#include <pvkernel/core/PVLRUCache.h>

#include <squey/PVPlotted.h>

#include <QString>

namespace PVRush
{
class PVNraw;
} // namespace PVRush

namespace Squey
{
class PVView;

class PVPlottedNrawCache
{
  public:
	/**
	 * Constructor
	 *
	 * @param view the related PVView
	 * @param col the tracked column
	 * @param the cache size
	 */
	PVPlottedNrawCache(const PVView& view, const PVCol col, const size_t size);

  public:
	/**
	 * Create the index if not created
	 */
	void initialize();

	/**
	 * Invalidate all cached entries
	 */
	void invalidate();

  public:
	/**
	 * Retrieve the nraw value associated with v
	 *
	 * @param v the plotting value
	 *
	 * @return nraw value associated to v
	 */
	const QString get(int64_t v);

  private:
	using value_type = PVPlotted::value_type;

	struct entry_t {
		entry_t() : plotted(0), row(0) {}
		entry_t(value_type p, PVRow r) : plotted(p), row(r) {}

		value_type plotted;
		PVRow row;

		bool operator<(const entry_t& rhs) const { return plotted < rhs.plotted; }
	};

	using entries_t = std::vector<entry_t>;

  private:
	PVCore::PVLRUCache<int64_t, QString> _cache;
	entries_t _entries;
	const Squey::PVView& _view;
	const PVRush::PVNraw& _nraw;
	PVCol _col;
};

} // namespace Squey

#endif // SQUEY_PVPLOTTEDNRAWCACHE_H
