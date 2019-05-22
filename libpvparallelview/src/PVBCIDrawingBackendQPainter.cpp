/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

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
	qDebug() << "PVBCIDrawingBackendQPainter::create_image(" << image_width << ", " << height_bits
	         << ")";
	// if (not _backend_image) {
	//     _backend_image = std::make_shared<PVBCIBackendImageQPainter>(image_width, height_bits);
	// }
	// return _backend_image;
	return std::make_shared<PVBCIBackendImageQPainter>(image_width, height_bits);
}

void PVParallelView::PVBCIDrawingBackendQPainter::
operator()(PVBCIBackendImage_p& backend_img,
           size_t x_start,
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
		const auto height = backend->height() * zoom_y;
		backend->pixmap() = QImage(width, height, QImage::Format_ARGB32);
		backend->pixmap().fill(Qt::transparent);

		QPainter painter(&backend->pixmap());

		// painter.fillRect(QRect{0, 0, backend->width(), backend->height()}, Qt::green);
		// painter.fillRect(QRect{0, 0, width, height}, Qt::red);
		qDebug() << "x_start:" << x_start << " width:" << width << " zoom_y:" << zoom_y;

		std::sort(codes, codes + n, [](PVBCICodeBase const& a, PVBCICodeBase const& b) {
			return a.as_10.int_v > b.as_10.int_v;
		});

		size_t valid_begin =
		    std::distance(codes, std::lower_bound(codes, codes + n, PVBCICodeBase{},
		                                          [](auto const& a, auto const& b) {
			                                          return a.as_10.int_v < PVROW_INVALID_VALUE;
			                                      }));

		auto start = std::chrono::steady_clock::now();

		if (height_bits == 10) {
			for (size_t i = valid_begin; i < n; ++i) {
				painter.setPen(PVCore::PVHSVColor(codes[i].as_10.s.color).toQColor());
				float left = codes[i].as_10.s.l / float(1 << height_bits);
				float right = codes[i].as_10.s.r / float(1 << height_bits);
				painter.drawLine(0, left * height, width, right * height);
			}
		} else {
			for (size_t i = valid_begin; i < n; ++i) {
				painter.setPen(PVCore::PVHSVColor(codes[i].as_11.s.color).toQColor());
				float left = codes[i].as_11.s.l / float(1 << height_bits);
				float right = codes[i].as_11.s.r / float(1 << height_bits);
				painter.drawLine(0, left * height, width, right * height);
			}
		}

		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> diff = end - start;
		std::cout << diff.count() << " s, lines:" << n - valid_begin << ", x_start:" << x_start
		          << ", total_pixel:" << width * height << "\n";

		render_done();
	});
	th.detach();
}