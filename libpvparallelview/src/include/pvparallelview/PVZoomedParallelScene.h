/**
 * \file PVZoomedParallelScene.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVZOOMEDPARALLELSCENE_H
#define PVPARALLELVIEW_PVZOOMEDPARALLELSCENE_H

#include <picviz/PVView_types.h>
#include <picviz/PVAxesCombination.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVFuncObserver.h>
#include <pvhive/PVCallHelper.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVSlidersManager_types.h>
#include <pvparallelview/PVSlidersGroup.h>
#include <pvparallelview/PVZoneRenderingBCI.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVZoomedParallelView.h>

#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsSceneWheelEvent>

#include <QObject>
#include <QDialog>
#include <QPaintEvent>
#include <QTimer>

#include <tbb/atomic.h>

namespace PVParallelView
{

// forward declaration
class PVZoomedSelectionAxisSliders;
class PVZonesProcessor;
class PVZoomedParallelViewSelectionLine;

/**
 * @class PVZoomedParallelScene
 *
 * A derived class of QGraphicsScene to use when displaying a zoom view of parallel coordinates
 * representation of events.
 */
class PVZoomedParallelScene : public QGraphicsScene
{
Q_OBJECT

private:
	friend class zoom_sliders_update_obs;
	friend class zoom_sliders_del_obs;

private:
	constexpr static size_t bbits = PARALLELVIEW_ZZT_BBITS;
	constexpr static double bbits_alpha_scale = 1. / (1. + (bbits - 10));

	constexpr static int axis_half_width = PARALLELVIEW_AXIS_WIDTH / 2;
	constexpr static uint32_t image_width = 512;
	constexpr static uint32_t image_height = 1024;

	constexpr static int zoom_steps = 5;
	constexpr static double root_step = pow(2.0, 1.0 / zoom_steps);

	// to make 2 consecutive values be separated by 1 pixel
	constexpr static int max_zoom_value = (32 - bbits);
	// to permit 2 consecutive values to be separated by 2^N pixels
	constexpr static int extra_zoom = 0; // actually inactive
	constexpr static int max_wheel_value = (max_zoom_value + extra_zoom) * zoom_steps;

private:
	typedef PVParallelView::PVZoomedZoneTree::context_t zzt_context_t;

public:
	typedef PVBCIBackendImage_p backend_image_p_t;

public:
	/**
	 * Constructor
	 *
	 * @param zpview the container/layout
	 * @param pvview_sp a shared pointer on the corresponding Picviz::PVView
	 * @param sliders_manager_p a shared pointer on the sliders manager
	 * @param zp_sel the PVZonesProcessor used for selection image
	 * @param zp_bg the PVZonesProcessor used for background image
	 * @param zm the corresponding zones manager
	 * @param axis_index the axis index this scene zoom on

	 */
	PVZoomedParallelScene(PVParallelView::PVZoomedParallelView *zpview,
	                      Picviz::PVView_sp& pvview_sp,
	                      PVSlidersManager_p sliders_manager_p,
	                      PVZonesProcessor& zp_sel,
	                      PVZonesProcessor& zp_bg,
	                      PVZonesManager const& zm,
	                      PVCol axis_index);

	/**
	 * Destructor
	 */
	~PVZoomedParallelScene();

	/**
	 * Overloaded methods when a mouse button is pressed.
	 *
	 * Control "drag & drop" panning and selection tool.
	 *
	 * @param event the pressed button event
	 */
	void mousePressEvent(QGraphicsSceneMouseEvent *event);

	/**
	 * Overloaded methods when a mouse button is released.
	 *
	 * Control "drag & drop" panning and selection tool.
	 *
	 * @param event the released button event
	 */
	void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);

	/**
	 * Overloaded methods when the mouse cursor moves.
	 *
	 * Control "drag & drop" panning and selection tool.
	 *
	 * @param event the movement event
	 */
	void mouseMoveEvent(QGraphicsSceneMouseEvent *event);

	/**
	 * Overloaded method for zooming and panning.
	 *
	 * @param event the wheel event
	 */
	void wheelEvent(QGraphicsSceneWheelEvent* event);

	/**
	 * Overloaded method when a key is pressed.
	 *
	 * lone key is "Enter" to force redraw.
	 *
	 * @param event the key pressed event
	 */
	void keyPressEvent(QKeyEvent *event);

	void update(const QRectF &rect = QRectF());

	/**
	 * Start asynchronously an update of selection rendering.
	 */
	void update_new_selection_async();
	/**
	 * Start asynchronously an update of selection and background rendering.
	 */
	void update_all_async();

	/**
	 * Update the zoomed view relatively to its PVView's axes combination.
	 */
	bool update_zones();

	/**
	 * Getter of the axis index used by this view.
	 */
	PVCol get_axis_index() const
	{
		return _axis_index;
	}

	/**
	 * Enable/disable the scene.
	 *
	 * In some case, the scene must be updated without having any side effect (Qt event,
	 * Hive propagation, new rendering in background, etc.). This method helps isolating the
	 * scene.
	 *
	 * @param value the state to apply
	 */
	void set_enabled(bool value)
	{
		if (!value) {
			cancel_and_wait_all_rendering();
		}
		_zpview->setDisabled(!value);
	}

	/**
	 * Overloaded method to draw the background.
	 *
	 * Principally: background color, axis, etc.
	 *
	 * @param painter the used QPainter
	 * @param rect the (unused) area to redraw
	 */
	virtual void drawBackground(QPainter *painter, const QRectF &rect);

	/**
	 * Function to call when the view is resized.
	 */
	void resize_display(bool need_recomputation = true)
	{
		update_zoom(need_recomputation);
	}

	/**
	 * Test if the zone \z has been rendered or not.
	 *
	 * @param z the zone id to test
	 *
	 * @return true if \z has been rendered; false otherwise
	 */

	inline bool is_zone_rendered(PVZoneID z) const
	{
		bool ret = false;
		if (_left_zone) {
			ret |= (z == _axis_index-1);
		}
		if (_right_zone) {
			ret |= (z == _axis_index);
		}
		return ret;
	}

	/**
	 * Prepare the zoomed view to be deleted.
	 */
	inline void about_to_be_deleted()
	{
		_view_deleted = true;
		_zpview->setDisabled(true);
		cancel_and_wait_all_rendering();
	}

protected:
	bool show_bg() const { return _show_bg; }

private slots:
	/**
	 * Start an update of the selection images.
	 */
	inline void update_sel()
	{
		_render_type = RENDER_SEL;
		update_display();
	}
	/**
	 * Start an update of the selection and background images.
	 */
	inline void update_all()
	{
		_render_type = RENDER_ALL;
		update_display();
	}

	/**
	 * Start an update of images given the internal state _render_type.
	 *
	 * must never be called directly, use ::update_all() or ::update_sel()
	 */
	void update_display();

	/**
	 * Update the graphical elements after a change of zoom parameter (mouse event or through the hive).
	 */
	void update_zoom(bool need_recomputation = true);

	/**
	 * Stops all pending rendering and wait for their ends.
	 */
	void cancel_and_wait_all_rendering();

private:
	/**
	 * Get the zoom level corresponding to the current mouse wheel state.
	 */
	inline int get_zoom_level()
	{
		return _wheel_value / zoom_steps;
	}

	/**
	 * Get the zoom step corresponding to the current mouse wheel state.
	 */
	inline int get_zoom_step()
	{
		return _wheel_value % zoom_steps;
	}

	/**
	 * Get the scale factor corresponding to the current mouse wheel state.
	 */
	inline double get_scale_factor()
	{
		// Phillipe's magic formula: 2^n × a^k
		return pow(2, get_zoom_level()) * pow(root_step, get_zoom_step());
	}

	/**
	 * Get the mouse wheel state from zoom level.
	 *
	 * This method is used by updater when the corresponding zoom sliders has been changed.
	 *
	 * @param a the desired scale factor
	 */
	inline double retrieve_wheel_value_from_alpha(const double &a)
	{
		// non simplified formula is: log2(1/a) / log2(root_steps)
		return -zoom_steps * log2(a);
	}

	/**
	 * Get the left zone's identifier.
	 */
	inline PVZoneID left_zone_id() const
	{
		return (_left_zone) ? _axis_index-1 : PVZONEID_INVALID;
	}

	/**
	 * Get the right zone's identifier.
	 */
	inline PVZoneID right_zone_id() const
	{
		return (_right_zone) ? _axis_index : PVZONEID_INVALID;
	}

	/**
	 * Getter of the underlying VPZonesManager.
	 */
	PVZonesManager const& get_zones_manager() const { return _zm; }

	/**
	 * Getter of the shared selection.
	 */
	inline Picviz::PVSelection& real_selection() { return _pvview.get_real_output_selection(); }
	inline Picviz::PVSelection& layer_stack_output_selection() { return _pvview.get_layer_stack_output_layer().get_selection(); }

	/**
	 * Get the \z zone's PVZoomedZoneTree.
	 *
	 * @param z the zone
	 */
	inline PVZoomedZoneTree const& get_zztree(PVZoneID const z) { return _zm.get_zone_tree<PVZoomedZoneTree>(z); }

	/**
	 * Connect in Qt's sense the slot \slots to the PVZoneRendering \zr.
	 *
	 * @param zr the zone rendering to connect to
	 * @param slots the slot
	 */
	void connect_zr(PVZoneRenderingBCI<bbits>* zr, const char* slots);

	inline size_t qimage_height() const { return 1<<PARALLELVIEW_ZZT_BBITS; }

	/**
	 * recreate composed images
	 */
	void recreate_images();

private slots:
	/**
	 * The slot called when the vertical scrollbar's value has changed.
	 *
	 * @param value the new scrollbar's value
	 */
	void scrollbar_changed_Slot(qint64 value);

	/**
	 * The slot called when the update time has expired.
	 */
	void updateall_timeout_Slot();

	/**
	 * The slot called when the all pending rendering has been done.
	 *
	 * It recreates each zone composed image.
	 */
	void all_rendering_done();

	/**
	 * The slot called when the selection rectangle commit an update.
	 */
	void commit_volatile_selection_Slot();

	/**
	 * The slot called when one pending rendering has been done.
	 *
	 * It recreates each zone composed image.
	 * @param zr the PVZoneRendering corresponding to the finished rendering
	 * @param zone_id the zone id corresponding to the finished rendering
	 */
	void zr_finished(PVParallelView::PVZoneRendering_p zr, int zone_id);

	/**
	 * The slot called when background visibility has to be toggled.
	 */
	void toggle_unselected_zombie_visibility();

private:
	/**
	 * @class zoom_sliders_update_obs
	 *
	 * PVHive observer when a zoom sliders has been updated.
	 */
	class zoom_sliders_update_obs :
		public PVHive::PVFuncObserver<PVSlidersManager,
		                              FUNC(PVSlidersManager::update_zoom_sliders)>
	{
	public:
		zoom_sliders_update_obs(PVZoomedParallelScene *parent = nullptr) : _parent(parent)
		{}

		void update(arguments_deep_copy_type const& args) const;

	private:
		PVZoomedParallelScene *_parent;
	};

	/**
	 * @class zoom_sliders_del_obs
	 *
	 * PVHive observer when a zoom sliders is deleted.
	 */
	class zoom_sliders_del_obs :
		public PVHive::PVFuncObserver<PVSlidersManager,
		                              FUNC(PVSlidersManager::del_zoom_sliders)>
	{
	public:
		zoom_sliders_del_obs(PVZoomedParallelScene *parent = nullptr) : _parent(parent)
		{}

		void update(arguments_deep_copy_type const& args) const;

	private:
		PVZoomedParallelScene *_parent;
	};

private:
	typedef PVParallelView::PVSlidersManager::axis_id_t axis_id_t;

private:
	/**
	 * @enum render_t
	 *
	 * Type of rendering when an update is requested.
	 */
	typedef enum {
		RENDER_ALL,
		RENDER_SEL
	} render_t;

	/**
	 * @class zone_desc_t
	 *
	 * This class groups together all informations relevant for a zone: Backend images, Qt images,
	 * PVZoneRendering, etc.
	 */

	struct zone_desc_t
	{
		zone_desc_t():
			last_zr_sel(),
			last_zr_bg()
		{ }

		inline void cancel_last_sel()
		{
			// AG: that copy is important if we are multi-threading and another thread
			// cleans our object after the following "if"
			PVZoneRenderingBCIBase_p zr = last_zr_sel;
			if (zr) {
				zr->cancel();
			}
		}

		inline void cancel_last_bg()
		{
			PVZoneRenderingBCIBase_p zr = last_zr_bg;
			if (zr) {
				zr->cancel();
			}
		}

		void cancel_all()
		{
			cancel_last_sel();
			cancel_last_bg();
		}

		void cancel_and_wait_all()
		{
			PVZoneRenderingBCIBase_p zr = last_zr_sel;
			if (zr) {
				zr->cancel();
				zr->wait_end();
				last_zr_sel.reset();
			}

			zr = last_zr_bg;
			if (zr) {
				zr->cancel();
				zr->wait_end();
				last_zr_bg.reset();
			}
		}

		backend_image_p_t           bg_image;   // the image for unselected/zomby lines
		backend_image_p_t           sel_image;  // the image for selected lines
		QGraphicsPixmapItem        *item;       // the scene's element
		QPointF                     next_pos;   // the item position of the next rendering
		PVZoneRenderingBCI_p<bbits> last_zr_sel;
		PVZoneRenderingBCI_p<bbits> last_zr_bg;
	};

private:
	PVZoomedParallelView           *_zpview;
	Picviz::PVView&                 _pvview;
	PVSlidersManager_p              _sliders_manager_p;
	PVSlidersGroup                 *_sliders_group;
	zoom_sliders_update_obs         _zsu_obs;
	zoom_sliders_del_obs            _zsd_obs;
	PVCol                           _axis_index;
	axis_id_t                       _axis_id;
	PVZonesManager const&           _zm;

	// this flag helps not killing twice through the hive and the destructor
	bool                            _pending_deletion;

	// about mouse
	int                             _wheel_value;
	qint64                          _pan_reference_y;

	// about zones rendering/display
	zone_desc_t                    *_left_zone;
	zone_desc_t                    *_right_zone;
	qreal                           _next_beta;
	qreal                           _current_beta;
	uint32_t                        _last_y_min;
	uint32_t                        _last_y_max;
	bool                            _show_bg;

	// about rendering
	QTimer                          _updateall_timer;
	PVZonesProcessor&		_zp_sel;
	PVZonesProcessor&		_zp_bg;

	// about selection in the zoom view
	PVZoomedParallelViewSelectionLine *_sel_line;
	PVZoomedSelectionAxisSliders   *_selection_sliders;
	PVHive::PVActor<Picviz::PVView> _view_actor;

	// about rendering invalidation
	render_t                        _render_type;
	int                             _renderable_zone_number;

	tbb::atomic<bool>               _view_deleted;
};

}

#endif // PVPARALLELVIEW_PVZOOMEDPARALLELSCENE_H
