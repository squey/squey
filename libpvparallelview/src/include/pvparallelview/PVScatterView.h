/**
 * \file PVScatterView.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef __PVSCATTERVIEW_H__
#define __PVSCATTERVIEW_H__

#include <pvkernel/core/PVSharedPointer.h>

#include <pvparallelview/PVZoomableDrawingAreaWithAxes.h>
#include <pvparallelview/PVZoomConverterScaledPowerOfTwo.h>

class QPainter;

namespace Picviz
{

class PVView;
typedef PVCore::PVSharedPtr<PVView> PVView_sp;

}

namespace PVParallelView
{

template <int STEPS>
class PVScatterViewZoomConverter;

class PVSelectionSquare;
class PVZoneTree;

class PVScatterView : public PVZoomableDrawingAreaWithAxes
{
	Q_OBJECT

	constexpr static int zoom_steps = 5;

public:
	PVScatterView(
		const Picviz::PVView_sp &pvview_sp,
        const PVZoneTree &zt,
		QWidget *parent = nullptr
	);
	~PVScatterView();

public:
	void about_to_be_deleted();
	void update_new_selection_async();
	void update_all_async();

protected:
	void drawBackground(QPainter *painter, const QRectF &rect) override;

private slots:
	void draw_points(QPainter *painter, const QRectF& rect);

private:
	Picviz::PVView& _view;
	PVZoneTree const& _zt;
	bool _view_deleted;
	PVScatterViewZoomConverter<zoom_steps>     *_zoom_converter;
};

}

#endif // __PVSCATTERVIEW_H__
