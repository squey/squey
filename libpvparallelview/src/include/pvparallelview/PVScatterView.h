/**
 * \file PVScatterView.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef __PVSCATTERVIEW_H__
#define __PVSCATTERVIEW_H__

#include <pvkernel/core/PVSharedPointer.h>

#include <pvparallelview/PVZoomableDrawingAreaWithAxes.h>

class QPainter;

namespace Picviz
{

class PVView;
typedef PVCore::PVSharedPtr<PVView> PVView_sp;

}

namespace PVParallelView
{

class PVSelectionSquare;
class PVSelectionSquareScatterView;
class PVZoneTree;
class PVZonesManager;
class PVZoomConverter;

class PVScatterView : public PVZoomableDrawingAreaWithAxes
{
	Q_OBJECT

	constexpr static int zoom_steps = 5;

	// the "digital" zoom level (to space consecutive values)
	constexpr static int zoom_extra_level = 0;
	constexpr static int zoom_extra = zoom_extra_level * zoom_steps;
	// -22 because we want a scale factor of 1 when the view fits in a 1024x1024 window
	constexpr static int zoom_min = -22 * zoom_steps;

public:
	PVScatterView(
		const Picviz::PVView_sp &pvview_sp,
		PVZonesManager const& zm,
		PVCol const axis_index,
		QWidget *parent = nullptr
	);
	~PVScatterView();

public:
	void about_to_be_deleted();
	void update_new_selection_async();
	void update_all_async();

	inline Picviz::PVView& lib_view() { return _view; }

protected:
	void drawBackground(QPainter *painter, const QRectF &rect) override;

private slots:
	void draw_points(QPainter *painter, const QRectF& rect);

private:
	Picviz::PVView& _view;
	PVZoneTree const& _zt;
	bool _view_deleted;
	PVZoomConverter *_zoom_converter;
	PVSelectionSquareScatterView* _selection_square;
};

}

#endif // __PVSCATTERVIEW_H__
