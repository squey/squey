/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef __PVFULLPARALLELSCENE_h__
#define __PVFULLPARALLELSCENE_h__

#include <QFuture>
#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsSceneWheelEvent>

#include <sigc++/sigc++.h>

#include <inendi/PVAxis.h>

#include <pvparallelview/PVBCIBackendImage_types.h>
#include <pvparallelview/PVFullParallelViewSelectionRectangle.h>
#include <pvparallelview/PVAxisGraphicsItem.h>
#include <pvparallelview/PVFullParallelView.h>
#include <pvparallelview/PVLinesView.h>
#include <pvparallelview/PVSlidersManager.h>

#include <tbb/atomic.h>

namespace PVParallelView
{

class PVFullParallelScene : public QGraphicsScene, public sigc::trackable
{
	Q_OBJECT

	friend class PVFullParallelViewSelectionRectangle;
	friend class draw_zone_Observer;
	friend class draw_zone_sel_Observer;

  public:
	PVFullParallelScene(PVFullParallelView* full_parallel_view,
	                    Inendi::PVView& view_sp,
	                    PVParallelView::PVSlidersManager* sm_p,
	                    PVBCIDrawingBackend& backend,
	                    PVZonesManager const& zm,
	                    PVZonesProcessor& zp_sel,
	                    PVZonesProcessor& zp_bg);
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

	Inendi::PVView& lib_view() { return _lib_view; }
	Inendi::PVView const& lib_view() const { return _lib_view; }

	void set_enabled(bool value)
	{
		if (!value) {
			_lines_view.cancel_and_wait_all_rendering();
		}
		_full_parallel_view->setDisabled(!value);
	}

	void update_new_selection_async();
	void update_all_async();
	void update_number_of_zones_async();

	/**
	 * Reset the zones layout and viewport to the following way: zones are resized to try fit
	 * in the viewport (according to the view's width); if zones fit in view, they are centered;
	 * otherwise, they are aligned to the left
	 */
	void reset_zones_layout_to_default();

	QRectF axis_scene_bounding_box(int axis) const
	{
		QRectF ret = _axes[axis]->sceneBoundingRect();
		return ret;
	}
	size_t axes_count() const { return _axes.size(); }

	QRectF axes_scene_bounding_box() const;

  protected:
	/**
	 * recompute the selected event number and update the displayed statistics
	 */
	void update_selected_event_number();

  private Q_SLOTS:
	void update_new_selection();
	void update_all();
	void update_number_of_zones();
	void toggle_unselected_zombie_visibility();
	void axis_hover_entered(PVCombCol col, bool entered);

  private:
	void update_zones_position(bool update_all = true, bool scale = true);
	void translate_and_update_zones_position();

	void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
	void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
	void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;
	void wheelEvent(QGraphicsSceneWheelEvent* event) override;
	void helpEvent(QGraphicsSceneHelpEvent* event) override;
	void keyPressEvent(QKeyEvent* event) override;

	inline QPointF map_to_axis(PVZoneID zone_id, QPointF p) const
	{
		return _axes[zone_id]->mapFromScene(p);
	}
	inline QPointF map_from_axis(PVZoneID zone_id, QPointF p) const
	{
		return _axes[zone_id]->mapToScene(p);
	}
	QRect map_to_axis(PVZoneID zone_id, QRectF rect) const
	{
		QRect r = _axes[zone_id]->map_from_scene(rect);

		// top and bottom must be corrected according to the y zoom factor
		r.setTop(r.top() / _zoom_y);
		r.setBottom(r.bottom() / _zoom_y);

		const int32_t zone_width = _lines_view.get_zone_width(zone_id);
		if (r.width() + r.x() > zone_width) {
			r.setRight(zone_width - 1);
		}

		return r;
	}

	bool sliders_moving() const;

	void add_zone_image();
	void add_axis(PVZoneID const zone_id, int index = -1);

	inline PVBCIDrawingBackend& backend() const { return _lines_view.backend(); }

	size_t qimage_height() const;

  private Q_SLOTS:
	void update_zone_pixmap_bg(PVZoneID zone_id);
	void update_zone_pixmap_sel(PVZoneID zone_id);
	void update_zone_pixmap_bgsel(PVZoneID zone_id);
	void scale_zone_images(PVZoneID zone_id);

	void update_selection_from_sliders_Slot(PVCol nraw_col);
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
	void zr_sel_finished(PVParallelView::PVZoneRendering_p zr, PVZoneID zone_id);
	void zr_bg_finished(PVParallelView::PVZoneRendering_p zr, PVZoneID zone_id);

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

	Inendi::PVView& _lib_view;

	PVFullParallelView* _full_parallel_view;

	PVFullParallelViewSelectionRectangle _sel_rect;

	qreal _translation_start_x = 0.0;

	float _zoom_y;
	float _axis_length;

	PVSlidersManager* _sm_p;

	QTimer* _timer_render;

	tbb::atomic<bool> _view_deleted;

	bool _show_min_max_values;
};
} // namespace PVParallelView

#endif // __PVFULLPARALLELSCENE_h__
