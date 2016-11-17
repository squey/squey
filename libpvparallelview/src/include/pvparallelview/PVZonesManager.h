/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVZONESMANAGER_H
#define PVPARALLELVIEW_PVZONESMANAGER_H

#include <QObject>

#include <pvkernel/core/PVAlgorithms.h>

#include <inendi/PVPlotted.h>
#include <inendi/PVView.h>

#include <pvparallelview/PVZone.h>
#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/PVRenderingJob.h>

// Forward declarations
namespace Inendi
{
class PVSelection;
} // namespace Inendi

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
	explicit PVZonesManager(Inendi::PVView const& view);
	PVZonesManager(PVZonesManager const&) = delete;

  public:
	void update_all();
	void reset_axes_comb();
	std::vector<PVZoneID> update_from_axes_comb(std::vector<PVCol> const& ac);
	std::vector<PVZoneID> update_from_axes_comb(Inendi::PVView const& view);
	void update_zone(PVZoneID zone);
	void reverse_zone(PVZoneID zone);
	void add_zone(PVZoneID zone);

	QSet<PVZoneID> list_cols_to_zones(QSet<PVCol> const& cols) const;

	void request_zoomed_zone(PVZoneID zone);

  public:
	PVZoneTree const& get_zone_tree(PVZoneID z) const
	{
		assert(z < get_number_of_managed_zones());
		return _zones[z].ztree();
	}

	PVZoneTree& get_zone_tree(PVZoneID z)
	{
		assert(z < get_number_of_managed_zones());
		return _zones[z].ztree();
	}

	PVZoomedZoneTree const& get_zoom_zone_tree(PVZoneID z) const
	{
		assert(z < get_number_of_managed_zones());
		return _zones[z].zoomed_ztree();
	}

	PVZoomedZoneTree& get_zoom_zone_tree(PVZoneID z)
	{
		assert(z < get_number_of_managed_zones());
		return _zones[z].zoomed_ztree();
	}

	void filter_zone_by_sel(PVZoneID zone_id, const Inendi::PVSelection& sel);
	void filter_zone_by_sel_background(PVZoneID zone_id, const Inendi::PVSelection& sel);

  public:
	inline PVZoneID get_number_of_managed_zones() const
	{
		return _view.get_axes_combination().get_combination().size() - 1;
	}

  public:
	inline PVZoneProcessing get_zone_processing(PVZoneID const z) const
	{
		const auto& plotted = _view.get_parent<Inendi::PVPlotted>();
		const auto& ac = _view.get_axes_combination();

		return {_view.get_row_count(), plotted.get_column_pointer(ac.get_nraw_axis(PVCombCol(z))),
		        plotted.get_column_pointer(ac.get_nraw_axis(PVCombCol(z + 1)))};
	}

  protected:
	const Inendi::PVView& _view;
	// _axes_comb is copied to handle update once the axes_combination have been update in the view.
	std::vector<PVCol> _axes_comb;
	std::vector<PVZone> _zones;
};
} // namespace PVParallelView

#endif
