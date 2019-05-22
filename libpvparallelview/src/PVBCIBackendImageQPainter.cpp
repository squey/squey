/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

#include <pvparallelview/PVBCIBackendImageQPainter.h>

PVParallelView::PVBCIBackendImageQPainter::PVBCIBackendImageQPainter(const uint32_t width,
                                                                     const uint8_t height_bits)
    : PVBCIBackendImage(width, height_bits), _pixmap()
{
	//_pixmap.fill(Qt::cyan);
}

QImage PVParallelView::PVBCIBackendImageQPainter::qimage(size_t crop_height) const
{
	assert(crop_height <= PVBCIBackendImage::height());

	return _pixmap;
}
