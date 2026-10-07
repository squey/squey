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

#include <tbb/task_group.h>

namespace
{

/**
 * The order the OpenCL kernel settles crossing lines in when they are drawn by
 * density, pixel_value() in bci_z24.cl: the line with the lowest is drawn.
 */
template <size_t Bbits>
uint32_t density_order(PVParallelView::PVBCICode<Bbits> const& code)
{
	const uint32_t black = code.s.color == HSV_COLOR_BLACK.h();
	const uint32_t transparency = 255 - code.opacity();
	return black << 31 | transparency << 23 | (code.s.idx >> 17) << 8 | code.s.color;
}

} // namespace

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
                                                         bool density,
                                                         bool antialiased,
                                                         std::function<void()> const& render_done)
{
	_jobs.run([=] {
		auto backend = static_cast<backend_image_t*>(backend_img.get());
		const auto height_bits = backend->height_bits();
		const auto height = backend->height() * zoom_y +
		                    2; // FIXME: this +2 is a workaround for similarity with OpenCL
		QImage paint_image(width, height, QImage::Format_ARGB32);
		paint_image.fill(Qt::transparent);

		QPainter painter(&paint_image);
		if (antialiased) {
			painter.setRenderHint(QPainter::Antialiasing);
			// The OpenCL kernel centres rows on whole values, where QPainter puts
			// their edges.
			painter.translate(0, 0.5);
		}

		size_t valid_begin = 0;
		if (density) {
			std::sort(codes, codes + n,
			          [height_bits](PVBCICodeBase const& a, PVBCICodeBase const& b) {
				          return height_bits == 10 ? density_order(a.as_10) > density_order(b.as_10)
				                                   : density_order(a.as_11) > density_order(b.as_11);
			          });
			// The line drawn last replaces what it crosses, opacity included, as the
			// kernel keeps one line per pixel rather than blending them.
			painter.setCompositionMode(QPainter::CompositionMode_Source);
		} else {
			// The OpenCL kernel settles overlapping lines by keeping the lowest row
			// index, so here the lowest index has to be drawn last, over the others.
			// Sorting on the whole code instead ordered by position, which decides
			// nothing, and left the two backends disagreeing about what is on top.
			std::sort(codes, codes + n, [](PVBCICodeBase const& a, PVBCICodeBase const& b) {
				return a.as_10.s.idx > b.as_10.s.idx;
			});

			valid_begin =
			    std::distance(codes, std::lower_bound(codes, codes + n, PVBCICodeBase{},
			                                          [](auto const& a, auto const&) {
				                                          return a.as_10.int_v < PVROW_INVALID_VALUE;
			                                          }));
		}

		const int x1 = reverse ? width : 0;
		const int x2 = reverse ? 0 : width;

		int last_pen = -1;
		const auto use_pen = [&](uint8_t color, uint8_t opacity) {
			const int pen = color | opacity << 8;
			if (pen != last_pen) {
				QColor pen_color = PVCore::PVHSVColor(color).toQColor();
				pen_color.setAlpha(opacity);
				painter.setPen(pen_color);
				last_pen = pen;
			}
		};

		if (height_bits == 10) {
			for (size_t i = valid_begin; i < n; ++i) {
				use_pen(codes[i].as_10.s.color, density ? codes[i].as_10.opacity() : 255);
				float left = codes[i].as_10.s.l / float(1 << height_bits);
				float right = codes[i].as_10.s.r / float(1 << height_bits);
				painter.drawLine(x1, left * height, x2, right * height);
			}
		} else {
			for (size_t i = valid_begin; i < n; ++i) {
				use_pen(codes[i].as_11.s.color, density ? codes[i].as_11.opacity() : 255);
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
}

void PVParallelView::PVBCIDrawingBackendQPainter::wait_all() const
{
	_jobs.wait();
}
