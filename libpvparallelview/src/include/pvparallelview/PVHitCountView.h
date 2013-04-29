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
#include <pvparallelview/PVZoomConverterScaledPowerOfTwo.h>

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

template <int STEPS>
class PVHitCountViewZoomConverter;

class PVHitCountViewInteractor;

class PVHitCountView : public PVZoomableDrawingAreaWithAxes
{
	Q_OBJECT

	friend class PVHitCountViewInteractor;

	constexpr static int zoom_steps = 5;
	// the "digital" zoom level (to space consecutive values)
	constexpr static int x_zoom_extra_level = 8;
	constexpr static int x_zoom_extra = x_zoom_extra_level * zoom_steps;
	constexpr static int y_zoom_extra_level = 0;
	constexpr static int y_zoom_extra = y_zoom_extra_level * zoom_steps;
	// -22 because we want a scale factor of 1 when the view fits in a 1024x1024 window
	constexpr static int zoom_min = -22 * zoom_steps;

private:
	typedef PVZoomConverterScaledPowerOfTwo<zoom_steps> zoom_converter_t;

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
	void drawBackground(QPainter *painter, const QRectF &rect) override;

private:
	int get_x_zoom_max_limit(const uint64_t value = 1L << 32,
	                         const uint64_t max_value = 1L << 32) const;

	void reset_view();

	void draw_lines(QPainter *painter,
	                const int src_x, const int view_top,
	                const int offset, const double &ratio,
	                const double rel_y_scale,
	                const uint32_t *buffer);

	void draw_clamped_lines(QPainter *painter,
	                        const int x_min, const int x_max,
	                        const int src_x, const int view_top,
	                        const int offset,
	                        const double rel_y_scale,
	                        const uint32_t *buffer);

private slots:
	void do_zoom_change();
	void do_pan_change();
	void do_update_all();

private slots:
	void update_all();
	void update_sel();

private:
	Picviz::PVView&                              _pvview;
	PVCol                                        _axis_index;
	QTimer                                       _update_all_timer;

	PVHitGraphBlocksManager                      _hit_graph_manager;
	bool                                         _view_deleted;
	uint64_t                                     _max_count;
	uint32_t                                     _block_base_pos;
	int                                          _block_zoom_level;
	bool                                         _show_bg;

	PVHitCountViewZoomConverter<zoom_steps>     *_x_zoom_converter;
	PVZoomConverterScaledPowerOfTwo<zoom_steps>  _y_zoom_converter;

	PVZoomableDrawingAreaInteractor             *_my_inteactor;
	PVZoomableDrawingAreaInteractor             *_hcv_inteactor;
};

}

#endif // PVPARALLELVIEW_PVHITCOUNTVIEW_H
