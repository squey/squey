#include <pvparallelview/PVZoomedZoneTree.h>
#include <pvparallelview/PVScatterViewDataImpl.h>

void PVParallelView::PVScatterViewDataImpl::process_bg(ProcessParams const& params, tbb::task_group_context* ctxt)
{
	params.zzt.browse_bci_by_y1_y2(
			params.y1_min,
			params.y1_max,
			params.y2_min,
			params.y2_max,
			params.zoom,
			params.alpha,
			params.colors,
			image_bg().get_hsv_image(),
			ctxt
			);

	image_bg().convert_image_from_hsv_to_rgb();
}

void PVParallelView::PVScatterViewDataImpl::process_sel(ProcessParams const& params, Picviz::PVSelection const& sel, tbb::task_group_context* ctxt)
{
	params.zzt.browse_bci_by_y1_y2_sel(
			params.y1_min,
			params.y1_max,
			params.y2_min,
			params.y2_max,
			params.zoom,
			params.alpha,
			params.colors,
			image_sel().get_hsv_image(),
			sel,
			ctxt
			);

	image_sel().convert_image_from_hsv_to_rgb();
};
