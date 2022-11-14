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

#ifndef _PVPARALLELVIEW_PVSERIESVIEWZOOMER_H_
#define _PVPARALLELVIEW_PVSERIESVIEWZOOMER_H_

#include <QWidget>
#include <QLabel>
#include <QBasicTimer>

namespace Inendi
{
class PVRangeSubSampler;
}

namespace PVParallelView
{

class PVSeriesView;

class PVViewZoomer : public QWidget
{
	Q_OBJECT
  public:
	using zoom_f = long double;
	struct Zoom {
		zoom_f minX;
		zoom_f maxX;
		zoom_f minY;
		zoom_f maxY;

		zoom_f width() const { return maxX - minX; }
		zoom_f height() const { return maxY - minY; }
	};

	PVViewZoomer(QWidget* parent = nullptr);

	void zoom_in(QRectF zoom_in_rect);
	void zoom_in(QPointF center, bool rectangular, zoom_f zoom_factor);
	void zoom_out();
	void zoom_out(QPointF center);
	void zoom_out(QPointF center, bool rectangular, zoom_f zoom_factor);
	void reset_zoom();
	void reset_and_zoom_in(Zoom zoom);

	void move_zoom_by(QPoint offset);

	QRect normalized_zoom_rect(QRect zoomRect, bool rectangular) const;
	Zoom rect_to_zoom(QRectF const& rect) const;
	Zoom current_zoom() const { return _zoom_stack[_current_zoom_index]; }

	static Zoom clamp_zoom(Zoom zoom);

  Q_SIGNALS:
	void zoom_updated(Zoom zoom);

  protected:
	virtual void update_zoom(Zoom) {}

  private:
	void zoom_out(QPointF center, Zoom oldZoom, Zoom targetZoom);
	void update_zoom();

  private:
	std::vector<Zoom> _zoom_stack;
	size_t _current_zoom_index = 0;
};

class PVSeriesViewZoomer : public PVViewZoomer
{
	Q_OBJECT
  public:
	PVSeriesViewZoomer(PVSeriesView* child,
	                   Inendi::PVRangeSubSampler& sampler,
	                   QWidget* parent = nullptr);
	virtual ~PVSeriesViewZoomer() = default;

	enum class SelectorMode { CrossHairs = 0, Zooming = 1, Selecting = 2, Hunting = 3 };

	SelectorMode current_selector_mode() const { return _selector_mode; }
	void change_selector_mode(SelectorMode const mode);
	void disable_selecting_mode(bool disabled) // TODO : implement selection mode on splitted series
	{
		_selecting_mode_disabled = disabled;
		if (_selector_mode == SelectorMode::Selecting) {
			change_selector_mode(SelectorMode::CrossHairs);
		}
	}

	QColor get_selector_color(SelectorMode mode) const { return _selector_colors[size_t(mode)]; }
	void set_selector_color(SelectorMode mode, QColor color)
	{
		_selector_colors[size_t(mode)] = color;
	}

	int get_cross_hairs_radius() const { return _cross_hairs_radius; }
	void set_cross_hairs_radius(int radius) { _cross_hairs_radius = radius; }

  Q_SIGNALS:
	void selector_mode_changed(SelectorMode previous_mode, SelectorMode current_mode);
	void selection_commit(Zoom selection);
	void cursor_moved(QRect region);
	void hunt_commit(QRect region, bool addition);

  protected:
	void mousePressEvent(QMouseEvent*) override;
	void mouseReleaseEvent(QMouseEvent*) override;
	void mouseMoveEvent(QMouseEvent*) override;
	void enterEvent(QEnterEvent* event) override;
	void leaveEvent(QEvent* event) override;
	void keyPressEvent(QKeyEvent* event) override;
	void keyReleaseEvent(QKeyEvent* event) override;

	void wheelEvent(QWheelEvent*) override;

	void resizeEvent(QResizeEvent*) override;
	void timerEvent(QTimerEvent* event) override;

	void update_zoom(Zoom zoom) override;
	void update_chronotips(QPoint point);
	void update_chronotips(QRect rect);

  private:
	void update_selector_and_chronotips();
	void update_selector_geometry(bool rectangular);
	void update_cross_hairs_geometry(QPoint pos);
	void update_chronotip_geometry(size_t chrono_index, QPoint pos);
	template <class T>
	void show_fragments(T const& fragments) const;
	template <class T>
	void hide_fragments(T const& fragments) const;
	QRect cross_hairs_rect(QPoint pos) const;

  private:
	PVSeriesView* _series_view;
	Inendi::PVRangeSubSampler& _rss;

	QBasicTimer _resizing_timer;

	SelectorMode _selector_mode = SelectorMode::CrossHairs;
	QRect _selector_rect;
	std::array<QWidget*, 4> _selector_fragments{nullptr};
	std::array<QColor, 4> _selector_colors{
	    QColor(255, 100, 50, 255), // CrossHairs
	    QColor(255, 0, 0, 255),    // Zooming
	    QColor(20, 255, 50, 255),  // Selecting
	    QColor(20, 20, 255, 255)   // Hunting
	};
	int _cross_hairs_radius = 10;
	std::array<QLabel*, 4> _chronotips{nullptr};

	bool _control_modifier = false;
	bool _left_button_down = false;

	bool _moving = false;
	QPoint _move_start;

	QTimer* _animation_timer;

	const zoom_f _centered_zoom_factor = 0.8;
	bool _selecting_mode_disabled = false;
};
} // namespace PVParallelView

#endif // _PVPARALLELVIEW_PVSERIESVIEWZOOMER_H_
