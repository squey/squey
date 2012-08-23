/**
 * \file PVZonesManager.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVZONESMANAGER_H
#define PVPARALLELVIEW_PVZONESMANAGER_H

#include <QObject>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVAlgorithms.h>

#include <picviz/PVPlotted.h>
#include <picviz/PVView_types.h>

#include <pvparallelview/PVZone.h>
#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/PVRenderingJob.h>

#include <boost/utility.hpp>

// Forward declarations
namespace Picviz {
class PVView;
class PVSelection;
}

namespace PVParallelView {

namespace __impl {
class ZoneCreation;
}

class PVZonesManager: public QObject, boost::noncopyable
{
	Q_OBJECT;

	friend class PVParallelView::__impl::ZoneCreation;

	typedef std::vector<PVZone> list_zones_t;
	typedef tbb::enumerable_thread_specific<PVZoneTree::ProcessData> process_ztree_tls_t;

public:
	PVZonesManager();

public:
	void update_all();
	void reset_axes_comb();
	void update_from_axes_comb(QVector<PVCol> const& ac);
	void update_from_axes_comb(Picviz::PVView const& view);
	void update_zone(PVZoneID zone);
	void reverse_zone(PVZoneID zone);
	void add_zone(PVZoneID zone);

public:
	template <class Tree>
	inline Tree const& get_zone_tree(PVZoneID z) const
	{
		assert(z < get_number_zones());
		return _zones[z].get_tree<Tree>();
	}

	template <class Tree>
	inline Tree& get_zone_tree(PVZoneID z)
	{
		assert(z < get_number_zones());
		return _zones[z].get_tree<Tree>();
	}

	inline uint32_t get_zone_width(PVZoneID z) const
	{
		assert(z < get_number_zones());
		return _zones[z].width();
	}

	void set_zone_width(PVZoneID zid, uint32_t width)
	{
		_zones[zid].set_width(PVCore::clamp(width, (uint32_t) PVParallelView::ZoneMinWidth, (uint32_t) PVParallelView::ZoneMaxWidth));
	}

	template <class F>
	void set_zones_width(F const& f)
	{
		for (PVZoneID zid = 0; zid < (PVZoneID) _zones.size(); zid++) {
			set_zone_width(zid, f(get_zone_width(zid)));
		}
	}

	void invalidate_selection();

	uint32_t get_zone_absolute_pos(PVZoneID z) const;
	PVZoneID get_zone_id(int abs_pos) const;

	bool filter_zone_by_sel(PVZoneID zid, const Picviz::PVSelection& sel);

public:
	void set_uint_plotted(Picviz::PVPlotted::uint_plotted_table_t const& plotted, PVRow nrows, PVCol ncols);
	void set_uint_plotted(Picviz::PVView const& view);
	inline PVZoneID get_number_zones() const { return _axes_comb.size()-1; }
	inline PVCol get_number_cols() const { return _ncols; }
	inline PVRow get_number_rows() const { return _nrows; }

protected:
	inline void get_zone_cols(PVZoneID z, PVCol& a, PVCol& b)
	{
		assert(z < get_number_zones());
		a = _axes_comb[z];
		b = _axes_comb[z+1];
	}
	inline Picviz::PVPlotted::uint_plotted_table_t const& get_uint_plotted() const { assert(_uint_plotted); return *_uint_plotted; }

signals:
	void filter_by_sel_finished(int zid);

protected:
	Picviz::PVPlotted::uint_plotted_table_t const* _uint_plotted = NULL;
	PVRow _nrows = 0;
	PVCol _ncols = 0;
	QVector<PVCol> _axes_comb;
	list_zones_t _zones;
	process_ztree_tls_t _tls_ztree;
	//list_zones_tree_t _quad_trees;
};



}

#endif
