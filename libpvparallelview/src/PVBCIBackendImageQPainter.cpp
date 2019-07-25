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
}

QImage PVParallelView::PVBCIBackendImageQPainter::qimage([[maybe_unused]] size_t crop_height) const
{
	assert(crop_height <= PVBCIBackendImage::height());

	return _pixmap;
}
