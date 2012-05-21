#ifndef PVPARALLELVIEW_PVZONESMANAGER_H
#define PVPARALLELVIEW_PVZONESMANAGER_H

#include <pvkernel/core/general.h>

#include <picviz/PVPlotted.h>
#include <picviz/PVView_types.h>

#include <pvparallelview/PVZoneTree.h>

// Forward declarations
namespace Picviz {
class PVView;
}

namespace PVParallelView {

class PVView;
class PVZoneQuadTree;

class PVZonesManager
{
	typedef std::vector<PVZoneTree> list_zones_tree_t;
//	typedef std::vector<PVZoneQuadTree> list_zones_quad_tree_t;

public:
	typedef PVCol PVZoneID;

public:
	PVZonesManager();

public:
	void update_all();
	void update_from_axes_comb(QVector<PVCol> const& ac);
	void update_from_axes_comb(Picviz::PVView const& view);
	void update_zone(PVZoneID zone);
	void reverse_zone(PVZoneID zone);
	void add_zone(PVZoneID zone);

public:
	void set_uint_plotted(Picviz::PVPlotted::uint_plotted_table_t const& plotted, PVRow nrows, PVCol ncols);
	void set_uint_plotted(Picviz::PVView const& view);

private:
	inline PVZoneID get_number_zones() const { return _axes_comb.size()-1; }
	inline PVRow get_number_rows() const { return _nrows; }
	inline void get_zone_cols(PVZoneID z, PVCol& a, PVCol& b)
	{
		assert(z < get_number_zones());
		a = _axes_comb[z];
		b = _axes_comb[z+1];
	}
	inline Picviz::PVPlotted::uint_plotted_table_t const& get_uint_plotted() const { assert(_uint_plotted); return *_uint_plotted; }

	
private:
	Picviz::PVPlotted::uint_plotted_table_t const* _uint_plotted;
	PVRow _nrows;
	PVCol _ncols;
	QVector<PVCol> _axes_comb;
	list_zones_tree_t _full_trees;
	//list_zones_tree_t _quad_trees;
};

}

#endif
