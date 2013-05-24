/**
 * \file PVScatterViewImage.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

#include <pvparallelview/PVZoomedZoneTree.h>
#include <pvparallelview/PVScatterViewDataImpl.h>

void PVParallelView::PVScatterViewDataImpl::process_bg(ProcessParams const& params, PVScatterViewImage& image, tbb::task_group_context* ctxt) const
{
	process_image(params, image, nullptr, ctxt);
}

void PVParallelView::PVScatterViewDataImpl::process_sel(ProcessParams const& params, PVScatterViewImage& image, Picviz::PVSelection const& sel, tbb::task_group_context* ctxt) const
{
	process_image(params, image, &sel, ctxt);
}

void PVParallelView::PVScatterViewDataImpl::process_image(
	ProcessParams const& params,
	PVScatterViewImage& image,
	Picviz::PVSelection const* sel,
	tbb::task_group_context* ctxt)
{
	if (params.can_optimize_translation()) { // In case of optimizable translation, recompute only the dirty rectangles
		PVLOG_INFO("OPTIMIZATION\n");
		// y2 translation
		if (params.y2_offset != 0)
		{
			const ProcessParams::dirty_rect& drect = params.rect_2();
			const QRect& drect_view = params.map_to_view(drect);

			PVCore::PVHSVColor* img = &image.get_hsv_image()[drect_view.y()*PVScatterViewImage::image_width+drect_view.x()];

			if (sel) {
				params.zzt->browse_bci_by_y1_y2_sel(
					drect.y1_min,
					drect.y1_max,
					drect.y2_min,
					drect.y2_max,
					params.zoom,
					params.alpha,
					params.colors,
					img,
					*sel,
					PVScatterViewImage::image_width,
					ctxt
				);
			}
			else {
				params.zzt->browse_bci_by_y1_y2(
					drect.y1_min,
					drect.y1_max,
					drect.y2_min,
					drect.y2_max,
					params.zoom,
					params.alpha,
					params.colors,
					img,
					PVScatterViewImage::image_width,
					ctxt
				);
			}
		}

		// y1 translation
		if (params.y1_offset != 0)
		{
			const ProcessParams::dirty_rect& drect = params.rect_1();
			const QRect& drect_view = params.map_to_view(drect);

			PVCore::PVHSVColor* img = &image.get_hsv_image()[drect_view.y()*PVScatterViewImage::image_width+drect_view.x()];

			if (sel) {
				params.zzt->browse_bci_by_y1_y2_sel(
					drect.y1_min,
					drect.y1_max,
					drect.y2_min,
					drect.y2_max,
					params.zoom,
					params.alpha,
					params.colors,
					img,
					*sel,
					PVScatterViewImage::image_width,
					ctxt
				);
			}
			else {
				params.zzt->browse_bci_by_y1_y2(
					drect.y1_min,
					drect.y1_max,
					drect.y2_min,
					drect.y2_max,
					params.zoom,
					params.alpha,
					params.colors,
					img,
					PVScatterViewImage::image_width,
					ctxt
				);
			}
		}
	}
	else { // In case of zoom, recompute the whole image
		PVLOG_INFO("NO OPTIMIZATION\n");
		if (sel) {
			params.zzt->browse_bci_by_y1_y2_sel(
				params.y1_min,
				params.y1_max,
				params.y2_min,
				params.y2_max,
				params.zoom,
				params.alpha,
				params.colors,
				image.get_hsv_image(),
				*sel,
				PVScatterViewImage::image_width,
				ctxt
			);
		}
		else {
			params.zzt->browse_bci_by_y1_y2(
				params.y1_min,
				params.y1_max,
				params.y2_min,
				params.y2_max,
				params.zoom,
				params.alpha,
				params.colors,
				image.get_hsv_image(),
				PVScatterViewImage::image_width,
				ctxt
			);
		}
	}
	
	image.convert_image_from_hsv_to_rgb();
}
