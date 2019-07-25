/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

#ifndef PVPARALLELVIEW_PVBCIBACKENDIMAGEQPAINTER_H
#define PVPARALLELVIEW_PVBCIBACKENDIMAGEQPAINTER_H

#include <pvparallelview/PVBCIBackendImage.h>

#include <QImage>
#include <QPixmap>

namespace PVParallelView
{

class PVBCIBackendImageQPainter : public PVParallelView::PVBCIBackendImage
{
	using pixel_t = uint32_t;

  public:
	PVBCIBackendImageQPainter(const uint32_t width, const uint8_t height_bits);

  public:
	QImage qimage(size_t crop_height) const override;

	QImage& pixmap() { return _pixmap; }

  private:
	QImage _pixmap;
};

} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVBCIBACKENDIMAGEQPAINTER_H
