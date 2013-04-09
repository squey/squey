
#ifndef PVPARALLELVIEW_PVZOOMABLEDRAWINGAREACONSTRAINTSMAJORY_H
#define PVPARALLELVIEW_PVZOOMABLEDRAWINGAREACONSTRAINTSMAJORY_H

#include <pvparallelview/PVZoomableDrawingAreaConstraints.h>

namespace PVParallelView
{

class PVZoomableDrawingAreaConstraintsMajorY : public PVZoomableDrawingAreaConstraints
{
	bool zoom_x_available() const override;

	bool zoom_y_available() const override;

	bool set_zoom_value(int axes, int value,
	                    PVParallelView::PVAxisZoom &zx,
	                    PVParallelView::PVAxisZoom &zy) override;

	bool increment_zoom_value(int axes, int value,
	                          PVParallelView::PVAxisZoom &zx,
	                          PVParallelView::PVAxisZoom &zy) override;

	void adjust_pan(QScrollBar64 *xsb, QScrollBar64 *ysb) override;
};

}

#endif // PVPARALLELVIEW_PVZOOMABLEDRAWINGAREACONSTRAINTSMAJORY_H
