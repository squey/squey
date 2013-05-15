/**
 * \file PVHitCountView.h
 *
 * Copyright (C) Picviz Labs 2010-2013
 */

#ifndef PVPARALLELVIEW_PVHITCOUNTVIEW_H
#define PVPARALLELVIEW_PVHITCOUNTVIEW_H

#include <pvkernel/core/PVSharedPointer.h>

#include <picviz/PVView.h>
#include <picviz/PVView_types.h>

#include <pvparallelview/PVZoomableDrawingAreaWithAxes.h>
#include <pvparallelview/PVHitGraphBlocksManager.h>
#include <pvparallelview/PVZoomConverterScaledPowerOfTwo.h>

#include <QTimer>
#include <QImage>

class QWidget;

namespace Picviz
{
class PVSelection;
}

namespace PVParallelView
{

class PVHitCountViewInteractor;
class PVSelectionRectangleHitCountView;
class PVSelectionRectangleInteractor;
class PVHitCountViewParamsWidget;

class PVHitCountView : public PVZoomableDrawingAreaWithAxes
{
	Q_OBJECT

	friend class PVHitCountViewInteractor;
	friend class PVHitCountViewParamsWidget;

	constexpr static int zoom_steps = 5;
	// the "digital" zoom level (to space consecutive values)
	constexpr static int y_zoom_extra_level = 10;
	constexpr static int y_zoom_extra = y_zoom_extra_level * zoom_steps;
	// to have a scale factor of 1 when the view fits in a 1024x1024 window (i.e. 2^22 value per pixel)
	constexpr static int y_min_zoom_level = 22;

	constexpr static int zoom_min = -y_min_zoom_level * zoom_steps;

	/* RH: nbits is 11, so that, the max level before needing a
	 * digital zoom is 21 instead of 22
	 */
	constexpr static int digital_zoom_level = y_min_zoom_level - 1;

private:
	typedef PVZoomConverterScaledPowerOfTwo<zoom_steps> zoom_converter_t;

public:
	PVHitCountView(const Picviz::PVView_sp &pvview_sp,
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

	inline uint32_t get_max_count() const { return _max_count; }

	inline Picviz::PVView& lib_view() { return _pvview; }

	inline const PVHitGraphBlocksManager& get_hit_graph_manager() const
	{
		return _hit_graph_manager;
	}

	inline PVHitGraphBlocksManager& get_hit_graph_manager()
	{
		return _hit_graph_manager;
	}

protected:
	void drawBackground(QPainter *painter, const QRectF &rect) override;
	void drawForeground(QPainter *painter, const QRectF &rect) override;

	void set_x_axis_zoom();
	void set_x_zoom_level_from_sel();

	inline int32_t get_x_zoom_min() const
	{
		return x_zoom_converter().scale_to_zoom((double)get_margined_viewport_width()/(double)_max_count);
	}

	void set_params_widget_position();

	inline Picviz::PVSelection& real_selection() { return _pvview.get_real_output_selection(); }
	inline Picviz::PVSelection& layer_stack_output_selection() { return _pvview.get_layer_stack_output_layer().get_selection(); }

	inline bool auto_x_zoom_sel() const { return _auto_x_zoom_sel; }
	inline bool show_bg() const { return _show_bg; }

protected slots:
	void toggle_auto_x_zoom_sel();
	void toggle_show_bg();

private:
	void reset_view();

	void draw_lines(QPainter *painter,
	                const int x_max,
	                const int block_view_offset,
	                const double rel_y_scale,
	                const uint32_t *buffer);

private:
	PVZoomConverterScaledPowerOfTwo<zoom_steps>&       x_zoom_converter()       { return _x_zoom_converter; }
	PVZoomConverterScaledPowerOfTwo<zoom_steps> const& x_zoom_converter() const { return _x_zoom_converter; }

	PVZoomConverterScaledPowerOfTwo<zoom_steps>&       y_zoom_converter()       { return _y_zoom_converter; }
	PVZoomConverterScaledPowerOfTwo<zoom_steps> const& y_zoom_converter() const { return _y_zoom_converter; }

	PVHitCountViewParamsWidget* params_widget() { return _params_widget; }

private slots:
	void do_zoom_change(int axes);
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
	int                                          _block_zoom_value;
	bool                                         _show_bg;
	bool                                         _auto_x_zoom_sel;
	bool                                         _do_auto_scale;
	
	PVZoomConverterScaledPowerOfTwo<zoom_steps>  _x_zoom_converter;
	PVZoomConverterScaledPowerOfTwo<zoom_steps>  _y_zoom_converter;

	PVZoomableDrawingAreaInteractor             *_my_interactor;
	PVZoomableDrawingAreaInteractor             *_hcv_interactor;

	PVSelectionRectangleHitCountView            *_sel_rect;
	PVSelectionRectangleInteractor              *_sel_rect_interactor;

	PVHitCountViewParamsWidget                  *_params_widget;
};

}

#endif // PVPARALLELVIEW_PVHITCOUNTVIEW_H
