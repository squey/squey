#include <pvparallelview/PVHitGraphDataInterface.h>


//
// PVHitGraphData
//

PVParallelView::PVHitGraphDataInterface::PVHitGraphDataInterface()
{
	buffer_all().set_zero();
	buffer_sel().set_zero();
}

PVParallelView::PVHitGraphDataInterface::~PVHitGraphDataInterface()
{
}

void PVParallelView::PVHitGraphDataInterface::shift_left(int n)
{
	buffer_all().shift_left(n);
	buffer_sel().shift_left(n);
}

void PVParallelView::PVHitGraphDataInterface::shift_right(int n)
{
	buffer_all().shift_right(n);
	buffer_sel().shift_right(n);
}

void PVParallelView::PVHitGraphDataInterface::process_allandsel(ProcessParams const& params, Picviz::PVSelection const& sel)
{
	process_all(params);
	process_sel(params, sel);
}
