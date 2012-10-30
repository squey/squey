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
#include <pvparallelview/PVSelectionSquareGraphicsItem.h>
#include <pvparallelview/PVZoneRendering.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVZoomedParallelView.h>

#include <QGraphicsPixmapItem>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsSceneWheelEvent>

#include <QObject>
#include <QDialog>
#include <QPaintEvent>
#include <QTimer>

namespace PVParallelView
{

class PVZoomedParallelView;
class PVZoomedSelectionAxisSliders;
class PVZonesProcessor;

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
	constexpr static int max_wheel_value = 21 * zoom_steps;

private:
	typedef PVParallelView::PVZoomedZoneTree::context_t zzt_context_t;

public:
	typedef PVBCIBackendImage_p backend_image_p_t;

public:
	PVZoomedParallelScene(PVParallelView::PVZoomedParallelView *zpview,
	                      Picviz::PVView_sp& pvview_sp,
	                      PVSlidersManager_p sliders_manager_p,
						  PVZonesProcessor& zp_sel,
						  PVZonesProcessor& zp_bg,
						  PVZonesManager const& zm,
	                      PVCol axis_index);

	~PVZoomedParallelScene();

	void mousePressEvent(QGraphicsSceneMouseEvent *event);
	void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
	void mouseMoveEvent(QGraphicsSceneMouseEvent *event);

	void wheelEvent(QGraphicsSceneWheelEvent* event);

	void keyPressEvent(QKeyEvent *event);

	void update_new_selection_async();
	void update_all_async();

	bool update_zones();

	PVCol get_axis_index() const
	{
		return _axis_index;
	}

	void set_enabled(bool value)
	{
		_zpview->setEnabled(value);
	}

	virtual void drawBackground(QPainter *painter, const QRectF &rect);

	void resize_display();

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

private slots:
	inline void update_sel()
	{
		_render_type = RENDER_SEL;
		update_display();
	}
	inline void update_all()
	{
		_render_type = RENDER_ALL;
		update_display();
	}

	// must not be called directly, use ::update_all() or ::update_sel()
	void update_display();

	void update_zoom();

private:
	inline int get_zoom_level()
	{
		return _wheel_value / zoom_steps;
	}

	inline int get_zoom_step()
	{
		return _wheel_value % zoom_steps;
	}

	inline double get_scale_factor()
	{
		// Phillipe's magic formula: 2^n × a^k
		return pow(2, get_zoom_level()) * pow(root_step, get_zoom_step());
	}

	inline double retrieve_wheel_value_from_alpha(const double &a)
	{
		// non simplified formula is: log2(1/a) / log2(root_steps)
		return -zoom_steps * log2(a);
	}

	inline PVZoneID left_zone_id() const
	{
		return (_left_zone) ? _axis_index-1 : PVZONEID_INVALID;
	}

	inline PVZoneID right_zone_id() const
	{
		return (_right_zone) ? _axis_index : PVZONEID_INVALID;
	}

	PVZonesManager const& get_zones_manager() const { return _zm; }

	inline Picviz::PVSelection& real_selection() { return _pvview.get_real_output_selection(); }

	inline PVZoomedZoneTree const& get_zztree(PVZoneID const z) { return _zm.get_zone_tree<PVZoomedZoneTree>(z); }

	void connect_zr(PVZoneRendering<bbits>* zr, const char* slots);

private slots:
	void scrollbar_changed_Slot(int value);
	void updateall_timeout_Slot();
	void all_rendering_done();
	void commit_volatile_selection_Slot();

	void zr_finished(int zid);

private:
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
	typedef enum {
		RENDER_ALL,
		RENDER_SEL
	} render_t;

	struct zone_desc_t
	{
		zone_desc_t():
			last_zr_sel(nullptr),
			last_zr_bg(nullptr)
		{ }

		inline void cancel_last_sel()
		{
			if (last_zr_sel) {
				last_zr_sel->cancel();
				//last_zr_sel->wait_end();
			}
		}

		inline void cancel_last_bg()
		{
			if (last_zr_bg) {
				last_zr_bg->cancel();
				//last_zr_bg->wait_end();
			}
		}

		void cancel_all()
		{
			cancel_last_sel();
			cancel_last_bg();
		}

		backend_image_p_t       bg_image;   // the image for unselected/zomby lines
		backend_image_p_t       sel_image;  // the image for selected lines
		//zzt_context_t           context;    // the extraction context for ZZT
		QGraphicsPixmapItem    *item;       // the scene's element
		QPointF                 next_pos;   // the item position of the next rendering
		PVZoneRendering<bbits>* last_zr_sel;
		PVZoneRendering<bbits>* last_zr_bg;
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
	int                             _pan_reference_y;

	// about zones rendering/display
	zone_desc_t                    *_left_zone;
	zone_desc_t                    *_right_zone;
	qreal                           _next_beta;
	qreal                           _current_beta;
	uint32_t                        _last_y_min;
	uint32_t                        _last_y_max;

	// about rendering
	QTimer                          _updateall_timer;
	PVZonesProcessor&				_zp_sel;
	PVZonesProcessor&				_zp_bg;

	// about selection in the zoom view
	QPointF                         _selection_rect_pos;
	PVSelectionSquareGraphicsItem  *_selection_rect;
	PVZoomedSelectionAxisSliders   *_selection_sliders;
	PVHive::PVActor<Picviz::PVView> _view_actor;

	// about rendering invalidation
	render_t                        _render_type;
	int                             _renderable_zone_number;
};

}

#endif // PVPARALLELVIEW_PVZOOMEDPARALLELSCENE_H
