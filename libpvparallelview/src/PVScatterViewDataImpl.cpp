/**
 * \file PVScatterViewImage.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

#include <pvparallelview/PVScatterViewDataImpl.h>

void PVParallelView::PVScatterViewDataImpl::process_image(
	ProcessParams const& params,
	PVScatterViewImage& image,
	Picviz::PVSelection const* sel /* = nullptr*/)
{
	if (params.y1_offset != 0 || params.y2_offset != 0) { // In case of translation, recompute only the dirty rectangles

		// y2 translation
		if (params.y2_offset != 0)
		{
			const ProcessParams::dirty_rect& drect = params.rect_2();
			const QRect& drect_view = params.map_to_view(drect);

			PVCore::PVHSVColor* img = &image.get_hsv_image()[drect_view.y()*PVScatterViewImage::image_width+drect_view.x()];

			if (sel) {
				params.zzt.browse_bci_by_y1_y2_sel(
					drect.y1_min,
					drect.y1_max,
					drect.y2_min,
					drect.y2_max,
					params.zoom,
					params.alpha,
					params.colors,
					img,
					PVScatterViewImage::image_width,
					*sel
				);
			}
			else {
				params.zzt.browse_bci_by_y1_y2(
					drect.y1_min,
					drect.y1_max,
					drect.y2_min,
					drect.y2_max,
					params.zoom,
					params.alpha,
					params.colors,
					img,
					PVScatterViewImage::image_width
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
				params.zzt.browse_bci_by_y1_y2_sel(
					drect.y1_min,
					drect.y1_max,
					drect.y2_min,
					drect.y2_max,
					params.zoom,
					params.alpha,
					params.colors,
					img,
					PVScatterViewImage::image_width,
					*sel
				);
			}
			else {
				params.zzt.browse_bci_by_y1_y2(
					drect.y1_min,
					drect.y1_max,
					drect.y2_min,
					drect.y2_max,
					params.zoom,
					params.alpha,
					params.colors,
					img,
					PVScatterViewImage::image_width
				);
			}
		}
	}
	else { // In case of zoom, recompute the whole image
		if (sel) {
			params.zzt.browse_bci_by_y1_y2_sel(
				params.y1_min,
				params.y1_max,
				params.y2_min,
				params.y2_max,
				params.zoom,
				params.alpha,
				params.colors,
				image.get_hsv_image(),
				PVScatterViewImage::image_width,
				*sel
			);
		}
		else {
			params.zzt.browse_bci_by_y1_y2(
				params.y1_min,
				params.y1_max,
				params.y2_min,
				params.y2_max,
				params.zoom,
				params.alpha,
				params.colors,
				image.get_hsv_image(),
				PVScatterViewImage::image_width
			);
		}
	}

	image.convert_image_from_hsv_to_rgb();
}
