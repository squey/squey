/**
 * \file PVScatterView.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef __PVSCATTERVIEW_H__
#define __PVSCATTERVIEW_H__

#include <QTimer>

#include <pvkernel/core/PVSharedPointer.h>

#include <picviz/PVAxesCombination.h>

#include <pvparallelview/PVScatterViewImagesManager.h>
#include <pvparallelview/PVZoomableDrawingAreaWithAxes.h>
#include <pvparallelview/PVZoomConverterScaledPowerOfTwo.h>
#include <pvparallelview/PVZoneRendering_types.h>

class QPainter;

namespace Picviz
{

class PVView;
typedef PVCore::PVSharedPtr<PVView> PVView_sp;

}

namespace PVParallelView
{

class PVSelectionSquare;
class PVSelectionSquareScatterView;
class PVZoneTree;
class PVZoomedZoneTree;
class PVZonesManager;
class PVZoomConverter;

class PVScatterView : public PVZoomableDrawingAreaWithAxes
{
	Q_OBJECT;

	constexpr static int zoom_steps = 5;

	// the "digital" zoom level (to space consecutive values)
	constexpr static int zoom_extra_level = 0;
	constexpr static int zoom_extra = zoom_extra_level * zoom_steps;
	// -22 because we want a scale factor of 1 when the view fits in a 1024x1024 window
	constexpr static int zoom_min = -22 * zoom_steps;

	/*! \brief This class represent an image that has been rendered, with its
	 * associated scene and viewport rect.
	 */
	class RenderedImage: boost::noncopyable
	{
	public:
		/*! \brief Swap the stored image with a new rendered one.
		 */
		void swap(QImage const& img, QRectF const& scene_rect, QRectF const& viewport_rect);

		/*! \brief Draw the image thanks to \a painter.
		 *
		 * This function assumes that \a painter uses the margined viewport coordinate system.
		 */
		void draw(PVGraphicsView* view, QPainter* painter);
	private:
		QRectF _scene_rect;
		QRectF _viewport_rect;
		QImage _img;
	};

public:
	PVScatterView(
		const Picviz::PVView_sp &pvview_sp,
		PVZonesManager const& zm,
		PVCol const axis_index,
		PVZonesProcessor& zp_bg,
		PVZonesProcessor& zp_sel,
		QWidget *parent = nullptr
	);
	~PVScatterView();

public:
	void about_to_be_deleted();
	void update_new_selection_async();
	void update_all_async();

	inline Picviz::PVView& lib_view() { return _view; }
	inline Picviz::PVView const& lib_view() const { return _view; }

	PVZoneID get_zone_index() const { return get_images_manager().get_zone_index(); }

	bool update_zones();

	void set_enabled(bool en);

public:
	static void toggle_show_quadtrees() { _show_quadtrees = !_show_quadtrees; }

protected:
	void drawBackground(QPainter *painter, const QRectF &rect) override;
	void keyPressEvent(QKeyEvent* event) override;

private slots:
	void do_update_all();
	void update_all();
	void update_sel();

	void update_img_bg(PVParallelView::PVZoneRendering_p zr, int zid);
	void update_img_sel(PVParallelView::PVZoneRendering_p zr, int zid);

private:
	inline PVZonesManager const& get_zones_manager() const { return get_images_manager().get_zones_manager(); }
	inline PVScatterViewImagesManager& get_images_manager() { return _images_manager; }
	inline PVScatterViewImagesManager const& get_images_manager() const { return _images_manager; }
	PVZoneTree const& get_zone_tree() const;

	void set_scatter_view_zone(PVZoneID const zid);

private slots:
	void do_zoom_change(int axes);
	void do_pan_change();

private:
	Picviz::PVView& _view;
	PVScatterViewImagesManager _images_manager;
	bool _view_deleted;
	PVZoomConverterScaledPowerOfTwo<zoom_steps> *_zoom_converter;
	PVSelectionSquareScatterView* _selection_square;
	static bool _show_quadtrees;

	RenderedImage _image_sel;
	RenderedImage _image_bg;

	QRectF _last_image_margined_viewport;
	QRectF _last_image_scene;

	Picviz::PVAxesCombination::axes_comb_id_t _axis_id;
};

}

#endif // __PVSCATTERVIEW_H__
