#ifndef PVPARALLELVIEW_PVZONESMANAGER_H
#define PVPARALLELVIEW_PVZONESMANAGER_H

#include <pvkernel/core/general.h>
#include <picviz/PVView_types.h>

// Forward declarations
namespace Picviz {
class PVView;
}

namespace PVParallelView {

class PVZoneTree;
class PVZoneQuadTree;

class PVZonesManager
{
	typedef std::vector<PVZoneTree> list_zones_tree_t;
//	typedef std::vector<PVZoneQuadTree> list_zones_quad_tree_t;

public:
	PVZonesManager(Picviz::PVView const& view);

public:
	void update_all();
	void update_zone(PVZoneID zone);
	void reverse_zone(PVZoneID zone);
	void add_zone(PVZoneID zone);

private:
	inline Picviz::PVAxesCombination const& get_axes_combination() const { return _view.get_axes_combination(); }
	inline PVZoneID get_number_zones() const { return get_axes_combination().get_axes_count(); }
	inline void get_zone_cols(PVZoneID z, PVCol& a, PVCol& b)
	{
		PVAxesCombination const& ac = get_axes_combination();
	}
	
private:
	Picviz::PVView const& _view;
	list_zones_tree_t _full_trees;
	//list_zones_tree_t _quad_trees;
};

}

#endif
