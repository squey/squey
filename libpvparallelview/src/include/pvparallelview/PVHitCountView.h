/**
 * \file PVHitCountView.h
 *
 * Copyright (C) Picviz Labs 2010-2013
 */

#ifndef PVPARALLELVIEW_PVHITCOUNTVIEW_H
#define PVPARALLELVIEW_PVHITCOUNTVIEW_H

#include <pvkernel/core/PVSharedPointer.h>

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
	QTransform scale_to_transform(const qreal x_scale_value,
	                              const qreal y_scale_value) const;

	QString get_y_value_at(const qint64 pos) const;

protected:
	void drawBackground(QPainter *painter, const QRectF &rect);
	void resizeEvent(QResizeEvent *event);
	void keyPressEvent(QKeyEvent *event);

private:
	void draw_lines(QPainter *painter,
	                const int src_x, const int view_top,
	                const int offset, const double &ratio,
	                const double rel_scale,
	                const uint32_t *buffer);

	void draw_clamped_lines(QPainter *painter,
	                        const int x_min, const int x_max,
	                        const int src_x, const int view_top,
	                        const int offset,
	                        const double rel_scale,
	                        const uint32_t *buffer);

private slots:
	void do_zoom_change();
	void do_pan_change();
	void do_update_all();

private slots:
	void update_all();
	void update_sel();

private:
	uint32_t                  _red_buffer[1024];
	const Picviz::PVView_sp  &_pvview_sp;
	const PVCol               _axis_index;
	QTimer                    _update_all_timer;

	PVHitGraphBlocksManager   _hit_graph_manager;
	bool                      _view_deleted;
	uint32_t                  _max_count;
	uint32_t                  _block_base_pos;
	int                       _block_zoom_level;
	bool                      _show_bg;
};

}

#endif // PVPARALLELVIEW_PVHITCOUNTVIEW_H
