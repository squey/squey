//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvparallelview/PVZoomedZoneTree.h>
#include <pvparallelview/PVScatterViewDataImpl.h>

void PVParallelView::PVScatterViewDataImpl::process_bg(ProcessParams const& params,
                                                       PVScatterViewImage& image,
                                                       tbb::task_group_context* ctxt) const
{
	process_image(params, image, nullptr, ctxt);
}

void PVParallelView::PVScatterViewDataImpl::process_sel(ProcessParams const& params,
                                                        PVScatterViewImage& image,
                                                        Squey::PVSelection const& sel,
                                                        tbb::task_group_context* ctxt) const
{
	process_image(params, image, &sel, ctxt);
}

void PVParallelView::PVScatterViewDataImpl::process_image(ProcessParams const& params,
                                                          PVScatterViewImage& image,
                                                          Squey::PVSelection const* sel,
                                                          tbb::task_group_context* ctxt)
{
	if (params.can_optimize_translation()) { // In case of optimizable
		                                     // translation, recompute only the
		                                     // dirty rectangles

		// y2 translation
		if (params.y2_offset != 0) {
			const ProcessParams::dirty_rect& drect = params.rect_2();
			const QRect& drect_view = params.map_to_view(drect);

			PVCore::PVHSVColor* img =
			    &image.get_hsv_image()[drect_view.y() * PVScatterViewImage::image_width +
			                           drect_view.x()];

			if (sel) {
				params.zzt->browse_bci_by_y1_y2_sel(
				    drect.y1_min, drect.y1_max, drect.y2_min, drect.y2_max, params.zoom,
				    params.alpha, params.colors, img, *sel, PVScatterViewImage::image_width, ctxt);
			} else {
				params.zzt->browse_bci_by_y1_y2(
				    drect.y1_min, drect.y1_max, drect.y2_min, drect.y2_max, params.zoom,
				    params.alpha, params.colors, img, PVScatterViewImage::image_width, ctxt);
			}
		}

		// y1 translation
		if (params.y1_offset != 0) {
			const ProcessParams::dirty_rect& drect = params.rect_1();
			const QRect& drect_view = params.map_to_view(drect);

			PVCore::PVHSVColor* img =
			    &image.get_hsv_image()[drect_view.y() * PVScatterViewImage::image_width +
			                           drect_view.x()];

			if (sel) {
				params.zzt->browse_bci_by_y1_y2_sel(
				    drect.y1_min, drect.y1_max, drect.y2_min, drect.y2_max, params.zoom,
				    params.alpha, params.colors, img, *sel, PVScatterViewImage::image_width, ctxt);
			} else {
				params.zzt->browse_bci_by_y1_y2(
				    drect.y1_min, drect.y1_max, drect.y2_min, drect.y2_max, params.zoom,
				    params.alpha, params.colors, img, PVScatterViewImage::image_width, ctxt);
			}
		}
	} else { // In case of zoom, recompute the whole image
		if (sel) {
			params.zzt->browse_bci_by_y1_y2_sel(params.y1_min, params.y1_max, params.y2_min,
			                                    params.y2_max, params.zoom, params.alpha,
			                                    params.colors, image.get_hsv_image(), *sel,
			                                    PVScatterViewImage::image_width, ctxt);
		} else {
			params.zzt->browse_bci_by_y1_y2(params.y1_min, params.y1_max, params.y2_min,
			                                params.y2_max, params.zoom, params.alpha, params.colors,
			                                image.get_hsv_image(), PVScatterViewImage::image_width,
			                                ctxt);
		}
	}

	image.convert_image_from_hsv_to_rgb();
}
