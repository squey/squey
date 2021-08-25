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

#ifndef __PVSCATTERVIEW_H__
#define __PVSCATTERVIEW_H__

#include <inendi/PVAxesCombination.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVScatterViewBackend.h>
#include <pvparallelview/PVZoomableDrawingAreaWithAxes.h>
#include <pvparallelview/PVZoomConverterScaledPowerOfTwo.h>
#include <pvparallelview/PVZoneRendering_types.h>

#include <boost/noncopyable.hpp>

#include <sigc++/sigc++.h>

#include <memory>

class QPainter;

namespace PVWidgets
{

class PVHelpWidget;
} // namespace PVWidgets

namespace Inendi
{

class PVView;
} // namespace Inendi

namespace PVParallelView
{

class PVZoneTree;
class PVZoomedZoneTree;
class PVZonesManager;
class PVZoomConverter;
class PVScatterViewParamsWidget;
class PVScatterViewInteractor;
class PVScatterViewSelectionRectangle;
class PVSelectionRectangleInteractor;

class PVScatterView : public PVZoomableDrawingAreaWithAxes, public sigc::trackable
{
	Q_OBJECT;

	friend PVScatterViewInteractor;
	friend PVScatterViewParamsWidget;

	constexpr static int zoom_steps = 5;

	// the "digital" zoom level (to space consecutive values)
	constexpr static int zoom_extra_level = 0;
	constexpr static int zoom_extra = zoom_extra_level * zoom_steps;
	// -22 because we want a scale factor of 1 when the view fits in a 1024x1024 window
	constexpr static int zoom_min = -22 * zoom_steps;

	/*! \brief This class represent an image that has been rendered, with its
	 * associated scene and viewport rect.
	 */
	class RenderedImage : boost::noncopyable
	{
	  public:
		/*! \brief Swap the stored image with a new rendered one.
		 */
		void swap(QImage const& img, QRectF const& viewport_rect, QTransform const& mv2s);

		/*! \brief Draw the image thanks to \a painter.
		 *
		 * This function assumes that \a painter uses the margined viewport coordinate system.
		 */
		void draw(PVGraphicsView* view, QPainter* painter);

	  private:
		QImage _img;
		QTransform _mv2s; // margined viewport to scene image transformation
	};

  public:
	using backend_unique_ptr_t = std::unique_ptr<PVScatterViewBackend>;
	using create_backend_t = std::function<backend_unique_ptr_t(PVZoneID, QWidget*)>;

	PVScatterView(Inendi::PVView& pvview_sp,
	              create_backend_t create_backend,
	              PVZoneID const zone_id,
	              QWidget* parent = nullptr);
	~PVScatterView() override;

  public:
	void about_to_be_deleted();
	void update_new_selection_async();
	void update_all_async();

	inline Inendi::PVView& lib_view() { return _view; }
	inline Inendi::PVView const& lib_view() const { return _view; }

	PVZoneID get_zone_id() const { return _zone_id; }

	bool update_zones();

	void set_enabled(bool en);

  public:
	static void toggle_show_quadtrees() { _show_quadtrees = !_show_quadtrees; }

  public:
	PVScatterViewSelectionRectangle* get_selection_rect() const { return _sel_rect.get(); }

  protected:
	void drawBackground(QPainter* painter, const QRectF& rect) override;
	void keyPressEvent(QKeyEvent* event) override;
	QString get_x_value_at(const qint64 value) override;
	QString get_y_value_at(const qint64 value) override;

  protected:
	void set_params_widget_position();

	PVWidgets::PVHelpWidget* help_widget() { return _help_widget; }

	bool show_bg() const { return _show_bg; }
	bool show_labels() const { return _show_labels; }

  private Q_SLOTS:
	void do_update_all();
	void update_all();
	void update_sel();

	void update_img_bg(PVParallelView::PVZoneRendering_p zr, PVZoneID zid);
	void update_img_sel(PVParallelView::PVZoneRendering_p zr, PVZoneID zid);

	void toggle_unselected_zombie_visibility();
	void toggle_show_labels();

  private:
	void update_labels_cache();

	PVZoneTree const& get_zone_tree() const;
	void set_scatter_view_zone(PVZoneID const zid);

	PVScatterViewParamsWidget* params_widget() { return _params_widget; }

  private Q_SLOTS:
	void do_zoom_change(int axes);
	void do_pan_change();

  private:
	Inendi::PVView& _view;
	backend_unique_ptr_t _backend;
	create_backend_t _create_backend;
	bool _view_deleted;
	std::unique_ptr<PVZoomConverterScaledPowerOfTwo<zoom_steps>> _zoom_converter;

	std::unique_ptr<PVZoomableDrawingAreaInteractor> _h_interactor;
	std::unique_ptr<PVZoomableDrawingAreaInteractor> _sv_interactor;

	std::unique_ptr<PVScatterViewSelectionRectangle> _sel_rect;
	std::unique_ptr<PVSelectionRectangleInteractor> _sel_rect_interactor;

	static bool _show_quadtrees;

	RenderedImage _image_sel;
	RenderedImage _image_bg;

	QRectF _last_image_margined_viewport;
	QTransform _last_image_mv2s;

	PVZoneID _zone_id;

	PVScatterViewParamsWidget* _params_widget;
	PVWidgets::PVHelpWidget* _help_widget;

	bool _show_bg;
	bool _show_labels;
};
} // namespace PVParallelView

#endif // __PVSCATTERVIEW_H__
