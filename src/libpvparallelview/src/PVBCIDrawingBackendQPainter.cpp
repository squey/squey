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

#include <pvparallelview/PVBCIDrawingBackendQPainter.h>
#include <pvparallelview/PVBCIBackendImageQPainter.h>
#include <pvkernel/core/PVHSVColor.h>

#include <QPainter>
#include <QDebug>
#include <thread>

PVParallelView::PVBCIDrawingBackendQPainter& PVParallelView::PVBCIDrawingBackendQPainter::get()
{
	static PVBCIDrawingBackendQPainter backend;
	return backend;
}

auto PVParallelView::PVBCIDrawingBackendQPainter::create_image(size_t image_width,
                                                               uint8_t height_bits)
    -> PVBCIBackendImage_p
{
	return std::make_shared<PVBCIBackendImageQPainter>(image_width, height_bits);
}

void PVParallelView::PVBCIDrawingBackendQPainter::render(PVBCIBackendImage_p& backend_img,
                                                         size_t /* x_start */,
                                                         size_t width,
                                                         PVBCICodeBase* codes,
                                                         size_t n,
                                                         const float zoom_y,
                                                         bool reverse,
                                                         std::function<void()> const& render_done)
{
	std::thread th([=] {
		auto backend = static_cast<backend_image_t*>(backend_img.get());
		const auto height_bits = backend->height_bits();
		const auto height = backend->height() * zoom_y +
		                    2; // FIXME: this +2 is a workaround for similarity with OpenCL
		QImage paint_image(width, height, QImage::Format_ARGB32);
		paint_image.fill(Qt::transparent);

		QPainter painter(&paint_image);

		std::sort(codes, codes + n, [](PVBCICodeBase const& a, PVBCICodeBase const& b) {
			return a.as_10.int_v > b.as_10.int_v;
		});

		size_t valid_begin =
		    std::distance(codes, std::lower_bound(codes, codes + n, PVBCICodeBase{},
		                                          [](auto const& a, auto const&) {
			                                          return a.as_10.int_v < PVROW_INVALID_VALUE;
		                                          }));

		const int x1 = reverse ? width : 0;
		const int x2 = reverse ? 0 : width;

		if (height_bits == 10) {
			for (size_t i = valid_begin; i < n; ++i) {
				painter.setPen(PVCore::PVHSVColor(codes[i].as_10.s.color).toQColor());
				float left = codes[i].as_10.s.l / float(1 << height_bits);
				float right = codes[i].as_10.s.r / float(1 << height_bits);
				painter.drawLine(x1, left * height, x2, right * height);
			}
		} else {
			for (size_t i = valid_begin; i < n; ++i) {
				painter.setPen(PVCore::PVHSVColor(codes[i].as_11.s.color).toQColor());
				if (codes[i].as_11.s.type == PVBCICode<11>::STRAIGHT) {
					float left = codes[i].as_11.s.l;
					float right = codes[i].as_11.s.r;
					painter.drawLine(x1, left * zoom_y, x2, right * zoom_y);
				} else if (codes[i].as_11.s.type == PVBCICode<11>::UP) {
					float left = codes[i].as_11.s.l;
					double right = codes[i].as_11.s.r;
					right = right + right / (2 * zoom_y * left);
					painter.drawLine(x1, left * zoom_y, reverse ? width - right : right, 0);
				} else if (codes[i].as_11.s.type == PVBCICode<11>::DOWN) {
					float left = codes[i].as_11.s.l;
					double right = codes[i].as_11.s.r;
					right = right - right / (2 * zoom_y * left);
					painter.drawLine(x1, left * zoom_y, reverse ? width - right : right, height);
				}
			}
		}

		backend->set_pixmap(std::move(paint_image));

		render_done();
	});
	th.detach();
}
