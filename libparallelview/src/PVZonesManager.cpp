#include <picviz/PVView.h>
#include <pvparallelview/PVZonesManager.h>

PVParallelView::PVZonesManager::PVZonesManager(Picviz::PVView const& view):
	_view(view)
{
}

void PVParallelView::PVZonesManager::update_all()
{
	PVZoneID nzones = get_number_zones();
	_full_trees.resize(nzones);
	//_quad_trees.resize(nzones);
	for (PVzoneID z = 0; z < nzones; z++) {
		_full_trees[z].set_trans_plotted();
	}
	
}
