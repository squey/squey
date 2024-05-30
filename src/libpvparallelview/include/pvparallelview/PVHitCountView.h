/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVPARALLELVIEW_PVHITCOUNTVIEW_H
#define PVPARALLELVIEW_PVHITCOUNTVIEW_H

#include <sigc++/sigc++.h>

#include <squey/PVView.h>

#include <pvparallelview/PVZoomableDrawingAreaWithAxes.h>
#include <pvparallelview/PVHitGraphBlocksManager.h>
#include <pvparallelview/PVZoomConverterScaledPowerOfTwo.h>
#include <pvparallelview/PVHitCountViewBackend.h>

#include <pvkernel/widgets/PVMouseButtonsLegend.h>

#include <QTimer>
#include <QSize>

#include <memory>

class QWidget;

namespace PVWidgets
{

class PVHelpWidget;
} // namespace PVWidgets

namespace Squey
{
class PVSelection;
} // namespace Squey

namespace PVParallelView
{

class PVHitCountViewInteractor;
class PVHitCountViewSelectionRectangle;
class PVSelectionRectangleInteractor;

class PVHitCountViewParamsWidget;

class PVHitCountView : public PVZoomableDrawingAreaWithAxes, public sigc::trackable
{
	Q_OBJECT

	friend class PVHitCountViewInteractor;
	friend class PVHitCountViewParamsWidget;

	constexpr static int zoom_steps = 5;
	// the "digital" zoom level (to space consecutive values)
	constexpr static int y_zoom_extra_level = 5;
	constexpr static int y_zoom_extra = y_zoom_extra_level * zoom_steps;
	// to have a scale factor of 1 when the view fits in a 1024x1024 window (i.e. 2^22 value per
	// pixel)
	constexpr static int y_min_zoom_level = 22;

	constexpr static int zoom_min = -y_min_zoom_level * zoom_steps;

	/* RH: nbits is 11, so that, the max level before needing a
	 * digital zoom is 21 instead of 22
	 */
	constexpr static int digital_zoom_level = y_min_zoom_level - 1;

  private:
	using zoom_converter_t = PVZoomConverterScaledPowerOfTwo<zoom_steps>;
	using backend_unique_ptr_t = std::unique_ptr<PVHitCountViewBackend>;
	using create_backend_t = std::function<backend_unique_ptr_t(PVCol, QWidget*)>;

  public:
	PVHitCountView(Squey::PVView& pvview_sp,
	               create_backend_t create_backend,
	               const PVCol axis,
	               QWidget* parent = nullptr);

	~PVHitCountView() override;

	QSize sizeHint() const override { return QSize(800, 200); }

  public:
	void about_to_be_deleted();
	void update_new_selection_async();
	void update_all_async();
	void update_all();
	bool update_zones();
	void set_enabled(const bool value);

	inline uint32_t get_max_count() const { return _max_count; }

	inline Squey::PVView& lib_view() { return _pvview; }
	inline Squey::PVView const& lib_view() const { return _pvview; }

	inline const PVHitGraphBlocksManager& get_hit_graph_manager() const
	{
		return _backend->get_hit_graph_manager();
	}

	inline PVHitGraphBlocksManager& get_hit_graph_manager()
	{
		return _backend->get_hit_graph_manager();
	}

	inline Squey::PVScaledNrawCache& get_y_labels_cache()
	{
		return _backend->get_y_labels_cache();
	}

	bool is_backend_valid() const
	{
		return (bool) _backend;
	}

  public:
	PVHitCountViewSelectionRectangle* get_selection_rect() const { return _sel_rect; }

  Q_SIGNALS:
	void set_status_bar_mouse_legend(PVWidgets::PVMouseButtonsLegend legend);
	void clear_status_bar_mouse_legend();

  protected:
	void drawBackground(QPainter* painter, const QRectF& rect) override;
	void drawForeground(QPainter* painter, const QRectF& rect) override;
	void enterEvent(QEnterEvent* event) override;
	void leaveEvent(QEvent* event) override;
	void keyPressEvent(QKeyEvent *event) override;
	void keyReleaseEvent(QKeyEvent *event) override;

	void set_x_axis_zoom();
	void set_x_zoom_level_from_sel();

	/**
	 * force an auto-scale update if auto-scale mode is activated)
	 */
	void request_auto_scale();

	inline int32_t get_x_zoom_min() const
	{
		return x_zoom_converter().scale_to_zoom((double)get_margined_viewport_width() /
		                                        (double)_max_count);
	}

	inline Squey::PVSelection const& real_selection() const
	{
		return _pvview.get_real_output_selection();
	}
	inline Squey::PVSelection& layer_stack_output_selection()
	{
		return _pvview.get_layer_stack_output_layer().get_selection();
	}

	inline bool auto_x_zoom_sel() const { return _auto_x_zoom_sel; }
	inline bool use_log_color() const { return _use_log_color; }
	inline bool show_labels() const { return _show_labels; }

	bool show_bg() const { return _show_bg; }

	void set_params_widget_position();
	QString get_x_value_at(const qint64 value) override;
	QString get_y_value_at(const qint64 value) override;

	void update_window_title(PVCol axis);

  protected Q_SLOTS:
	void toggle_auto_x_zoom_sel();
	void toggle_log_color();
	void toggle_show_labels();

  private:
	void reset_view();

	void draw_lines(QPainter* painter,
	                const int x_max,
	                const int block_view_offset,
	                const double rel_y_scale,
	                const uint32_t* buffer,
	                const int hsv_value);

  private:
	PVZoomConverterScaledPowerOfTwo<zoom_steps>& x_zoom_converter() { return _x_zoom_converter; }
	PVZoomConverterScaledPowerOfTwo<zoom_steps> const& x_zoom_converter() const
	{
		return _x_zoom_converter;
	}

	PVZoomConverterScaledPowerOfTwo<zoom_steps>& y_zoom_converter() { return _y_zoom_converter; }
	PVZoomConverterScaledPowerOfTwo<zoom_steps> const& y_zoom_converter() const
	{
		return _y_zoom_converter;
	}

	PVHitCountViewParamsWidget* params_widget() { return _params_widget; }
	PVWidgets::PVHelpWidget* help_widget() { return _help_widget; }

  private Q_SLOTS:
	void do_zoom_change(int axes);
	void do_pan_change();
	void do_update_all();

	void toggle_unselected_zombie_visibility();

  private Q_SLOTS:
	void update_sel();

  private:
	Squey::PVView& _pvview;
	QTimer _update_all_timer;

	backend_unique_ptr_t _backend;
	create_backend_t _create_backend;
	bool _view_deleted;
	uint64_t _max_count;
	int _block_zoom_value;
	bool _show_bg;
	bool _auto_x_zoom_sel;
	bool _do_auto_scale;
	bool _use_log_color;
	bool _show_labels;
	PVZoomConverterScaledPowerOfTwo<zoom_steps> _x_zoom_converter;
	PVZoomConverterScaledPowerOfTwo<zoom_steps> _y_zoom_converter;

	PVZoomableDrawingAreaInteractor* _my_interactor;
	PVZoomableDrawingAreaInteractor* _hcv_interactor;

	PVHitCountViewSelectionRectangle* _sel_rect;
	PVSelectionRectangleInteractor* _sel_rect_interactor;

	PVHitCountViewParamsWidget* _params_widget;
	PVWidgets::PVHelpWidget* _help_widget;

	PVWidgets::PVMouseButtonsLegend _mouse_buttons_current_legend;
	PVWidgets::PVMouseButtonsLegend _mouse_buttons_default_legend;
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVHITCOUNTVIEW_H
