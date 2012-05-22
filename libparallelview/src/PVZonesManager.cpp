#include <picviz/PVView.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVZoneProcessing.h>

PVParallelView::PVZonesManager::PVZonesManager():
	_uint_plotted(NULL)
{
}

void PVParallelView::PVZonesManager::set_uint_plotted(Picviz::PVPlotted::uint_plotted_table_t const& plotted, PVRow nrows, PVCol ncols)
{
	_uint_plotted = &plotted;
	_nrows = nrows;
	_ncols = ncols;

	// Init original axes combination
	_axes_comb.clear();
	_axes_comb.reserve(ncols);
	for (PVCol c = 0; c < ncols; c++) {
		_axes_comb.push_back(c);
	}
}

void PVParallelView::PVZonesManager::update_all()
{
	PVZoneID nzones = get_number_zones();
	PVLOG_INFO("(PVZonesManager::update_all) number of zones = %d\n", nzones);
	assert(nzones >= 1);
	_zones.resize(nzones);
	
	PVZoneProcessing zp(get_uint_plotted(), get_number_rows());
	for (PVZoneID z = 0; z < nzones; z++) {
		get_zone_cols(z, zp.col_a(), zp.col_b());
		PVZoneTree& ztree = _zones[z].ztree();
		ztree.process(zp);
	}
}

void PVParallelView::PVZonesManager::update_from_axes_comb(QVector<PVCol> const& ac)
{
	// TODO: optimise this to update only the concerned zones !
	_axes_comb = ac;
	update_all();
}

void PVParallelView::PVZonesManager::update_from_axes_comb(Picviz::PVView const& view)
{
	update_from_axes_comb(view.get_axes_combination().get_axes_index_list());
}

void PVParallelView::PVZonesManager::set_uint_plotted(Picviz::PVView const& view)
{
	set_uint_plotted(view.get_plotted_parent()->get_uint_plotted(), view.get_row_count(), view.get_column_count());
}

size_t PVParallelView::PVZonesManager::get_zone_absolute_pos(PVZoneID zone) const
{
	assert(z < _zones.size());
	size_t pos = 0;
	for (PVZoneID z = 0; z < zone; z++) {
		pos += _zones[z].width() + PVParallelView::AxisWidth;
	}
	return pos;
}
