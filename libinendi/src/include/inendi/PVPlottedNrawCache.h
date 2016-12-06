/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2016
 */

#ifndef INENDI_PVPLOTTEDNRAWCACHE_H
#define INENDI_PVPLOTTEDNRAWCACHE_H

#include <pvbase/types.h>

#include <pvkernel/core/PVLRUCache.h>

#include <inendi/PVPlotted.h>

#include <QString>

namespace PVRush
{
class PVNraw;
} // namespace PVRush

namespace Inendi
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
	const Inendi::PVView& _view;
	const PVRush::PVNraw& _nraw;
	PVCol _col;
};

} // namespace Inendi

#endif // INENDI_PVPLOTTEDNRAWCACHE_H
