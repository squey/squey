/**
 * \file PVHitCountView.h
 *
 * Copyright (C) Picviz Labs 2010-2013
 */

#ifndef PVPARALLELVIEW_PVHITCOUNTVIEW_H
#define PVPARALLELVIEW_PVHITCOUNTVIEW_H

#include <pvkernel/core/PVSharedPointer.h>

#include <picviz/PVSelection.h>

#include <pvparallelview/PVZoomableDrawingAreaWithAxes.h>
#include <pvparallelview/PVHitGraphBlocksManager.h>


#include <QTimer>
#include <QImage>

class QWidget;

namespace Picviz
{

class PVView;
typedef PVCore::PVSharedPtr<PVView> PVView_sp;

}

namespace PVParallelView
{

class PVZoneTree;

class PVHitCountView : public PVZoomableDrawingAreaWithAxes
{
	Q_OBJECT

	constexpr static int zoom_steps = 5;
	constexpr static double root_step = pow(2.0, 1.0 / zoom_steps);

public:
	PVHitCountView(const Picviz::PVView_sp &pvview_sp,
	               const PVZoneTree &zt,
	               const uint32_t *col_plotted,
	               const PVRow nrows,
	               const PVCol axis_index,
	               QWidget *parent = nullptr);

	~PVHitCountView();

public:
	void about_to_be_deleted();
	void update_new_selection_async();
	void update_all_async();
	void set_enabled(const bool value);

protected:
	qreal zoom_to_scale(const int zoom_value) const;
	int scale_to_zoom(const qreal scale_value) const;

	QString get_y_value_at(const qint64 pos) const;

protected:
	void drawBackground(QPainter *painter, const QRectF &rect);
	void resizeEvent(QResizeEvent *event);

private slots:
	void do_zoom_change();
	void do_pan_change();
	void do_update_all();

private slots:
	void update_all();
	void update_sel();

private:
	void recompute_back_buffer();

private:
	uint32_t                  _red_buffer[1024];
	const Picviz::PVView_sp  &_pvview_sp;
	const PVZoneTree         &_zt;
	const uint32_t           *_col_plotted;
	const PVRow               _nrows;
	const PVCol               _axis_index;
	QTimer                     _update_all_timer;
	uint32_t                  _back_image_pos;
	QImage                    _back_image;

	Picviz::PVSelection       _selection;
	PVHitGraphBlocksManager   _hit_graph_manager;
	bool                      _view_deleted;
};

}

#endif // PVPARALLELVIEW_PVHITCOUNTVIEW_H
