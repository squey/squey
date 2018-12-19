/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVZOOMABLEDRAWINGAREACONSTRAINTSHOMOTHETIC_H
#define PVPARALLELVIEW_PVZOOMABLEDRAWINGAREACONSTRAINTSHOMOTHETIC_H

#include <pvparallelview/PVZoomableDrawingAreaConstraints.h>

namespace PVParallelView
{

class PVZoomableDrawingAreaConstraintsHomothetic : public PVZoomableDrawingAreaConstraints
{
	bool zoom_x_available() const override;

	bool zoom_y_available() const override;

	bool set_zoom_value(int axes, int value, PVAxisZoom& zx, PVAxisZoom& zy) override;

	bool increment_zoom_value(int /*axes*/, int value, PVAxisZoom& zx, PVAxisZoom& zy) override;

	void adjust_pan(QScrollBar* xsb, QScrollBar* ysb) override;
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVZOOMABLEDRAWINGAREACONSTRAINTSHOMOTHETIC_H
