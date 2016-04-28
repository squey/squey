/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVZONESMANAGER_H
#define PVPARALLELVIEW_PVZONESMANAGER_H

#include <QObject>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVAlgorithms.h>

#include <inendi/PVPlotted.h>
#include <inendi/PVView_types.h>

#include <pvparallelview/PVZone.h>
#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/PVRenderingJob.h>

#include <boost/utility.hpp>

// Forward declarations
namespace Inendi
{
class PVView;
class PVSelection;
}

namespace PVParallelView
{

namespace __impl
{
class ZoneCreation;
}

class PVZonesManager : public QObject, boost::noncopyable
{
	friend class PVParallelView::__impl::ZoneCreation;

	typedef tbb::enumerable_thread_specific<PVZoneTree::ProcessData> process_ztree_tls_t;

  public:
	explicit PVZonesManager(Inendi::PVView const& view);

  public:
	typedef Inendi::PVAxesCombination::axes_comb_id_t axes_comb_id_t;
	typedef Inendi::PVAxesCombination::columns_indexes_t columns_indexes_t;

	void update_all();
	void reset_axes_comb();
	std::vector<PVZoneID> update_from_axes_comb(columns_indexes_t const& ac);
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
	inline PVZoneID get_number_of_managed_zones() const { return _axes_comb.size() - 1; }
	inline PVCol get_number_cols() const { return _ncols; }
	inline PVRow get_row_count() const { return _nrows; }

	inline Inendi::PVPlotted::uint_plotted_table_t const& get_uint_plotted() const
	{
		assert(_uint_plotted);
		return *_uint_plotted;
	}

  public:
	inline void get_zone_plotteds(PVZoneID const z, uint32_t const** plotted_a,
	                              uint32_t const** plotted_b) const
	{
		PVCol a, b;
		get_zone_cols(z, a, b);
		*plotted_a =
		    Inendi::PVPlotted::get_plotted_col_addr(get_uint_plotted(), get_row_count(), a);
		*plotted_b =
		    Inendi::PVPlotted::get_plotted_col_addr(get_uint_plotted(), get_row_count(), b);
	}

	inline void get_zone_cols(PVZoneID z, PVCol& a, PVCol& b) const
	{
		assert(z < get_number_of_managed_zones());
		a = _axes_comb[z].get_axis();
		b = _axes_comb[z + 1].get_axis();
	}

  protected:
	Inendi::PVPlotted::uint_plotted_table_t const* _uint_plotted =
	    nullptr;                  // FIXME : This is a duplication, it should get it from view
	PVRow _nrows = 0;             // FIXME : This is a duplication, it should get it from view
	PVCol _ncols = 0;             // FIXME : This is a duplication, it should get it from view
	columns_indexes_t _axes_comb; // FIXME : This is a duplication, it should get it from view
	std::vector<PVZone> _zones;
	process_ztree_tls_t _tls_ztree;
};
}

#endif
