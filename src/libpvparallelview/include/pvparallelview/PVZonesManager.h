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

#ifndef PVPARALLELVIEW_PVZONESMANAGER_H
#define PVPARALLELVIEW_PVZONESMANAGER_H

#include <unordered_map>
#include <unordered_set>

#include <QObject>

#include <pvkernel/core/PVAlgorithms.h>

#include <squey/PVScaled.h>
#include <squey/PVView.h>

#include <pvparallelview/PVZone.h>
#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/PVRenderingJob.h>

// Forward declarations
namespace Squey
{
class PVSelection;
} // namespace Squey

namespace PVParallelView
{

namespace __impl
{
class ZoneCreation;
} // namespace __impl

class PVZonesManager : public QObject
{
	friend class PVParallelView::__impl::ZoneCreation;

  public:
	explicit PVZonesManager(Squey::PVView const& view);
	PVZonesManager(PVZonesManager const&) = delete;

  public:
	class ZoneRetainer
	{
		friend class PVZonesManager;

	  private:
		ZoneRetainer(PVZonesManager& zm, PVZoneID zid) : zone_id(zid), _zm(zm)
		{
			_zm._zones_ref_count.insert(zone_id);
		}

	  public:
		ZoneRetainer(ZoneRetainer&& o) : zone_id(o.zone_id), _zm(o._zm)
		{
			o.zone_id = PVZONEID_INVALID;
		}
		~ZoneRetainer()
		{
			_zm._zones_ref_count.extract(zone_id);
			_zm.release_zone(zone_id);
		}

	  private:
		PVZoneID zone_id;
		PVZonesManager& _zm;
	};

  public:
	void update_all(bool reinit_zones = true);
	void reset_axes_comb();
	void update_from_axes_comb(std::vector<PVCol> const& ac);
	void update_from_axes_comb(Squey::PVView const& view);
	void update_zone(PVZoneID zone);
	[[nodiscard]] auto acquire_zone(PVZoneID zone) -> ZoneRetainer;
	void release_zone(PVZoneID zone);

	std::unordered_set<PVZoneID> list_cols_to_zones_indices(QSet<PVCombCol> const& comb_cols) const;

	void request_zoomed_zone(PVZoneID zone);

  public:
	PVZoneTree const& get_zone_tree(PVZoneID z) const { return get_zone(z).ztree(); }

	PVZoneTree& get_zone_tree(PVZoneID z) { return get_zone(z).ztree(); }

	PVZoomedZoneTree const& get_zoom_zone_tree(PVZoneID z) const
	{
		return get_zone(z).zoomed_ztree();
	}

	PVZoomedZoneTree& get_zoom_zone_tree(PVZoneID z) { return get_zone(z).zoomed_ztree(); }

	void filter_zone_by_sel(PVZoneID zone_id, const Squey::PVSelection& sel);
	void filter_zone_by_sel_background(PVZoneID zone_id, const Squey::PVSelection& sel);

  public:
	/* Get the number of managed zones from axes combination. Some zones are independant (e.g. a
	 * random scatter view) and thus not counted. */
	[[deprecated]] inline size_t get_number_of_managed_zones() const
	{
		return _view.get_axes_combination().get_combination().size() - 1;
	}

	size_t get_number_of_axes_comb_zones() const
	{
		auto sz = _view.get_axes_combination().get_combination().size();
		return sz >= 2 ? sz - 1 : 0;
	}
	size_t get_number_of_zones() const { return _zones.size(); }

	/* Get the index currently associated with the ZoneID. The index is invalidated at any call to
	 * `update_from_axes_comb`. */
	// size_t get_zone_index(PVZoneID z) const { return _zone_indices.find(z)->second; }
	std::unordered_set<size_t> get_zone_indices(PVZoneID z) const
	{
		auto pr = _zone_indices.equal_range(z);
		std::unordered_set<size_t> zi_set;
		std::for_each(pr.first, pr.second,
		              [&zi_set](auto zid_zix) { zi_set.insert(zid_zix.second); });
		return zi_set;
	}
	PVZoneID get_zone_id(size_t index) const { return _zones[index].first; }
	bool has_zone(PVZoneID z) const { return _zone_indices.count(z) > 0; }

  public:
	inline PVZoneProcessing get_zone_processing(PVZoneID const z) const
	{
		const auto& scaled = _view.get_parent<Squey::PVScaled>();

		return {_view.get_row_count(), scaled.get_column_pointer(z.first),
		        scaled.get_column_pointer(z.second)};
	}

  protected:
	const Squey::PVView& _view;
	// _axes_comb is copied to handle update once the axes_combination have been update in the view.
	std::vector<PVCol> _axes_comb;
	std::vector<std::pair<PVZoneID, PVZone>> _zones;
	std::unordered_multimap<PVZoneID, decltype(_zones)::size_type> _zone_indices;
	// reference counting for non-managed zones, works with ZoneRetainer.
	std::unordered_multiset<PVZoneID> _zones_ref_count;

  protected:
	PVZone& get_zone(PVZoneID z)
	{
		assert(_zone_indices.count(z) > 0);
		return _zones[_zone_indices.find(z)->second].second;
	}

	PVZone const& get_zone(PVZoneID z) const
	{
		assert(_zone_indices.count(z) > 0);
		return _zones[_zone_indices.find(z)->second].second;
	}
};
} // namespace PVParallelView

#endif
