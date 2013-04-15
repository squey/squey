
#ifndef PVPARALLELVIEW_PVZOOMABLEDRAWINGAREACONSTRAINTSHOMOTHETIC_H
#define PVPARALLELVIEW_PVZOOMABLEDRAWINGAREACONSTRAINTSHOMOTHETIC_H

#include <pvparallelview/PVZoomableDrawingAreaConstraints.h>

namespace PVParallelView
{

class PVZoomableDrawingAreaConstraintsHomothetic : public PVZoomableDrawingAreaConstraints
{
	bool zoom_x_available() const override;

	bool zoom_y_available() const override;

	bool set_zoom_value(int axes, int value,
	                    PVAxisZoom &zx, PVAxisZoom &zy) override;

	bool increment_zoom_value(int /*axes*/, int value,
	                          PVAxisZoom &zx, PVAxisZoom &zy) override;

	void adjust_pan(QScrollBar64 *xsb, QScrollBar64 *ysb) override;
};

}

#endif // PVPARALLELVIEW_PVZOOMABLEDRAWINGAREACONSTRAINTSHOMOTHETIC_H
