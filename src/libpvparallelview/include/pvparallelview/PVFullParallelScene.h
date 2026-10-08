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

#ifndef __PVFULLPARALLELSCENE_h__
#define __PVFULLPARALLELSCENE_h__

#include <QFuture>
#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsSceneWheelEvent>

#include <pvkernel/widgets/PVWheelEventAccumulator.h>

#include <sigc++/sigc++.h>

#include <squey/PVAxis.h>

#include <pvparallelview/PVBCIBackendImage_types.h>
#include <pvparallelview/PVFullParallelViewSelectionRectangle.h>
#include <pvparallelview/PVAxisGraphicsItem.h>
#include <pvparallelview/PVFullParallelView.h>
#include <pvparallelview/PVLinesView.h>
#include <pvparallelview/PVSlidersManager.h>
#include <pvparallelview/PVViewRenderingContext.h>

#include <atomic>
#include <optional>
#include <unordered_set>
#include <utility>

namespace PVParallelView
{

class PVViewRenderingContext;

class PVFullParallelScene : public QGraphicsScene, public sigc::trackable
{
	Q_OBJECT

	friend class PVFullParallelViewSelectionRectangle;
	friend class draw_zone_Observer;
	friend class draw_zone_sel_Observer;

  public:
	PVFullParallelScene(PVFullParallelView* full_parallel_view,
	                    Squey::PVView& view_sp,
	                    PVViewRenderingContext& context,
	                    PVBCIDrawingBackend& backend);
	~PVFullParallelScene() override;

	void first_render();
	void update_all_with_timer();
	void scale_all_zones_images();

	void update_viewport();
	void update_scene(bool recenter_view);

	void about_to_be_deleted();

	PVFullParallelView* graphics_view() { return _full_parallel_view; }

	PVParallelView::PVLinesView& get_lines_view() { return _lines_view; }
	PVParallelView::PVLinesView const& get_lines_view() const { return _lines_view; }

	Squey::PVView& lib_view() { return _lib_view; }
	Squey::PVView const& lib_view() const { return _lib_view; }

	/**
	 * Stop drawing, or draw again.
	 *
	 * Called as zones are rebuilt, from whichever thread rebuilds them: a scaling
	 * asked for from the GUI is computed in the thread of its progress box. The
	 * drawing is cancelled there and then, the widget -- which belongs to the GUI
	 * thread -- is switched from the GUI thread.
	 */
	void set_enabled(bool value)
	{
		if (!value) {
			_lines_view.cancel_and_wait_all_rendering();
		}
		QMetaObject::invokeMethod(
		    _full_parallel_view, [view = _full_parallel_view, value] { view->setDisabled(!value); },
		    Qt::AutoConnection);
	}

	void update_new_selection_async();
	void update_all_async();
	void update_all();
	void update_number_of_zones_async();
	void update_number_of_zones();

	/**
	 * Reset the zones layout and viewport to the following way: zones are resized to try fit
	 * in the viewport (according to the view's width); if zones fit in view, they are centered;
	 * otherwise, they are aligned to the left
	 */
	void reset_zones_layout_to_default();

	size_t axes_count() const { return _axes.size(); }

	QRectF axes_scene_bounding_box() const;

	void enable_density_on_axes(bool enable_density);

	/**
	 * Draw the lines by density, the line of a single row having @p opacity, in
	 * ]0, 1]; at 1, lines are opaque, as they are otherwise. See
	 * PVLinesView::set_line_opacity.
	 */
	void set_line_opacity(float opacity);
	float line_opacity() const { return _lines_view.get_line_opacity(); }

	/**
	 * Antialias the lines (see PVLinesView::set_antialiased).
	 */
	void set_antialiased(bool antialiased);
	bool is_antialiased() const { return _lines_view.is_antialiased(); }

	/**
	 * Selection scaling: spread the selection over the whole axes.
	 *
	 * The setting lives on the Squey::PVScaled, so it reaches every view built on
	 * the same scaling and is saved with the investigation. See
	 * Squey::PVScaled::set_scale_on_selection.
	 */
	void set_scale_on_selection(bool enabled);
	void set_auto_scale_on_selection(bool enabled);

	/**
	 * Rescale the axes over the rows currently selected.
	 */
	void rescale_on_selection();

  private:
	/**
	 * The band the selected rows occupy on an axis, in slider values.
	 *
	 * Empty when nothing is selected. The bounds are scaled values as the sliders
	 * hold them, which is also what a scene ordinate is derived from.
	 */
	std::optional<std::pair<int64_t, int64_t>> selection_band(PVCombCol col) const;

  public:

  protected:
	/**
	 * recompute the selected event number and update the displayed statistics
	 */
	void update_selected_event_number();

  private Q_SLOTS:
	void update_new_selection();

	/**
	 * Rescale on the selection when the scaling was told to follow it.
	 *
	 * Held back until the selection stops changing. A rectangle being dragged
	 * commits a selection every PVSelectionRectangle::delay_msec, and answering
	 * each one means rescaling every column and rebuilding every zone tree behind
	 * them, over and over, for selections nobody has looked at yet.
	 *
	 * Reached through a queued connection: rescaling emits the scaling's own
	 * update, which the rendering context answers by rebuilding zones, and that
	 * must not run inside the emission of the selection change that led here.
	 */
	void rescale_on_selection_if_automatic();
	void toggle_unselected_zombie_visibility();
	void axis_hover_entered(PVCombCol col, bool entered);

  private:
	// Rendering-context (PVViewRenderingContext) signal handlers
	//
	// Connected through sigc::mem_fun, which the scene being a sigc::trackable
	// disconnects as it is destroyed. A lambda capturing the scene is not, and the
	// context outlives every scene built on it -- one in a dock goes as the dock is
	// closed -- so the next emission would call into a scene that is gone.
	void on_selection_updated_rescale();
	void on_axes_combination_changed(bool async);
	void on_zones_about_to_be_updated(std::unordered_set<PVZoneID> const& zones);
	void on_zones_updated(std::unordered_set<PVZoneID> const& zones);
	void on_view_about_to_be_deleted();
	void on_context_about_to_be_deleted();

	/**
	 * Have the densities of the axes showing these columns drawn again.
	 *
	 * The columns are the scaling's, which the axes are not numbered by: an axis's
	 * position is its place in the combination, which may leave columns out, repeat
	 * them or reorder them.
	 *
	 * Connected through sigc::mem_fun, which the scene being a sigc::trackable
	 * disconnects as it is destroyed. A lambda capturing the scene is not, and the
	 * scaling outlives every scene built on it -- one in a dock goes as the dock is
	 * closed -- so the next rescaling would call into a scene that is gone.
	 */
	void refresh_densities(const QList<PVCol>& columns);

	void update_number_of_visible_zones();
	void update_zones_position(bool update_all = true, bool scale = true);
	void translate_and_update_zones_position();

	void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
	void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
	void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;
	void wheelEvent(QGraphicsSceneWheelEvent* event) override;
	void helpEvent(QGraphicsSceneHelpEvent* event) override;
	void keyPressEvent(QKeyEvent* event) override;

	inline QPointF map_to_axis(size_t zone_index, QPointF p) const
	{
		return _axes[zone_index]->mapFromScene(p);
	}
	QRect map_to_axis(size_t zone_index, QRectF rect) const
	{
		QRect r = _axes[zone_index]->map_from_scene(rect);

		// top and bottom must be corrected according to the y zoom factor
		r.setTop(r.top() / _zoom_y);
		r.setBottom(r.bottom() / _zoom_y);

		const int32_t total_zone_width =
		    _lines_view.get_zone_width(zone_index) + _lines_view.get_axis_width();
		if (r.width() + r.x() > total_zone_width) {
			r.setRight(total_zone_width - 1);
		}

		return r;
	}

	bool sliders_moving() const;

	void add_zone_image();
	void add_axis(size_t const zone_index, int index = -1);

	inline PVBCIDrawingBackend& backend() const { return _lines_view.backend(); }

	size_t qimage_height() const;

  private Q_SLOTS:
	void update_zone_pixmap_bgsel(size_t zone_index);
	void update_zone_pixmap_bg(size_t zone_index);
	void update_zone_pixmap_sel(size_t zone_index);
	void scale_zone_images(size_t zone_index);

	void update_selection_from_sliders_Slot(PVCombCol col);
	void scrollbar_pressed_Slot();
	void scrollbar_released_Slot();

	void highlight_axis(int col, bool entered);
	void sync_axis_with_section(size_t col, size_t pos);

	void emit_new_zoomed_parallel_view(PVCombCol axis_index)
	{
		Q_EMIT _full_parallel_view->new_zoomed_parallel_view(&_lib_view, axis_index);
	}

  private Q_SLOTS:
	// Slots called from PVLinesView
	void zr_sel_finished(PVParallelView::PVZoneRendering_p zr, PVZoneID zid);
	void zr_bg_finished(PVParallelView::PVZoneRendering_p zr, PVZoneID zid);
	void zr_sel_finished(PVParallelView::PVZoneRendering_p zr, size_t zone_index);
	void zr_bg_finished(PVParallelView::PVZoneRendering_p zr, size_t zone_index);

	void render_all_zones_all_imgs();

	void update_axes_layer_min_max();

  private:
	int32_t pos_last_axis() const;

  private:
	struct SingleZoneImagesItems {
		QGraphicsPixmapItem* sel;
		QGraphicsPixmapItem* bg;

		SingleZoneImagesItems() {}

		void setPos(QPointF point)
		{
			sel->setPos(point);
			bg->setPos(point);
		}

		void setPixmap(QPixmap const& pixmap_sel, QPixmap const& pixmap_bg)
		{
			sel->setPixmap(pixmap_sel);
			bg->setPixmap(pixmap_bg);
		}

		void hide()
		{
			sel->hide();
			bg->hide();
		}

		void show()
		{
			sel->show();
			bg->show();
		}

		void remove(QGraphicsScene* scene)
		{
			scene->removeItem(sel);
			scene->removeItem(bg);
		}
	};

  private:
	typedef std::vector<PVParallelView::PVAxisGraphicsItem*> axes_list_t;

  private:
	PVParallelView::PVLinesView _lines_view;

	std::vector<SingleZoneImagesItems> _zones;
	axes_list_t _axes;

	Squey::PVView& _lib_view;
	PVViewRenderingContext* _context;

	PVFullParallelView* _full_parallel_view;

	PVFullParallelViewSelectionRectangle _sel_rect;

	qreal _translation_start_x = 0.0;

	float _zoom_y;
	float _axis_length;

	PVSlidersManager* _sm_p;

	// Read from update_all() and update_new_selection(), which the model can reach
	// through queued calls: never left holding whatever was on the stack.
	QTimer* _timer_render = nullptr;
	QTimer* _timer_rescale = nullptr;

	// Set once this scene is detached from its rendering context or model
	// (teardown): pending render-finished callbacks must then be ignored.
	std::atomic<bool> _detached;

	bool _show_min_max_values;
	bool _density_on_axes_enabled = false;

	// Held while the lines are drawn by density.
	PVViewRenderingContext::RowCounting _row_counting;

	// Only zoom once per whole physical wheel notch (ignore high-resolution sub-notch events).
	PVWidgets::PVWheelEventAccumulator _wheel_accumulator;
};
} // namespace PVParallelView

#endif // __PVFULLPARALLELSCENE_h__
