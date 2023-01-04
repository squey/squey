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

#ifndef __PVPARALLELVIEW_PVAXISHEADER_H__
#define __PVPARALLELVIEW_PVAXISHEADER_H__

#include <pvbase/types.h>

#include <QGraphicsRectItem>
#include <QEasingCurve>
#include <QGraphicsSceneMouseEvent>

class QPropertyAnimation;
class QPainter;
class QGraphicsSceneMouseEvent;

namespace Inendi
{
class PVView;
} // namespace Inendi

namespace PVParallelView
{

class PVAxisGraphicsItem;
namespace __impl
{
class PVAxisSelectedAnimation;
} // namespace __impl

/**
 * Axis label highlight decoration
 *
 * @note as this class reimplements a single click using mouse press/release,
 * it can interfere with other graphics items. To avoid that problem, each
 * time the "click" mouse button is pressed, the current event is backed-up
 * and resend only when the next event can not lead to a "click" event: press-
 * release is a click but press-move-release is not.
 */

class PVAxisHeader : public QObject, public QGraphicsRectItem
{
	Q_OBJECT
  public:
	PVAxisHeader(const Inendi::PVView& view, PVCombCol comb_col, PVAxisGraphicsItem* parent);

  public:
	void set_width(int width);
	void start(bool start);

	PVAxisGraphicsItem* axis();
	PVAxisGraphicsItem const* axis() const;
	bool is_last_axis() const;

  protected:
	void hoverEnterEvent(QGraphicsSceneHoverEvent* event) override;
	void hoverLeaveEvent(QGraphicsSceneHoverEvent* event) override;
	void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
	void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;
	void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
	void contextMenuEvent(QGraphicsSceneContextMenuEvent* event) override;

  Q_SIGNALS:
	void mouse_hover_entered(PVCombCol col, bool entered);
	void mouse_clicked(PVCombCol col);
	void new_zoomed_parallel_view(PVCombCol axis_index);
	void new_selection_slider();
	void change_mapping(QString const& selected_plotting);
	void change_plotting(QString const& selected_plotting);

  private Q_SLOTS:
	void new_zoomed_parallel_view();

  private:
	PVCol get_axis_index() const;

  private:
	const Inendi::PVView& _view;
	PVCombCol _comb_col;

	__impl::PVAxisSelectedAnimation* _axis_selected_animation;
	bool _started = false;
	bool _clicked;
	QGraphicsSceneMouseEvent _click_event;
};

namespace __impl
{

class PVGraphicsPolygonItem;

class PVAxisSelectedAnimation : QObject
{
	Q_OBJECT

	Q_PROPERTY(qreal opacity READ get_opacity WRITE set_opacity);

  private:
	static constexpr qreal opacity_animation_min_amount = 0.2;
	static constexpr qreal opacity_animation_max_amount = 1.0;
	static constexpr size_t opacity_animation_duration_ms = 200;
	static constexpr QEasingCurve::Type opacity_animation_easing = QEasingCurve::Linear;

  public:
	explicit PVAxisSelectedAnimation(PVAxisHeader* parent);
	~PVAxisSelectedAnimation() override;

  public:
	void start(bool start);

  private:                                    // properties
	qreal get_opacity() const { return 0.0; } // avoid Qt warning
	void set_opacity(qreal opacity);

  private:
	inline PVAxisHeader* header() { return static_cast<PVAxisHeader*>(parent()); }

  private:
	QPropertyAnimation* _opacity_animation;
	QPropertyAnimation* _blur_animation;

	PVGraphicsPolygonItem* _title_highlight;
};

class PVGraphicsPolygonItem : public QGraphicsPolygonItem
{
	void paint(QPainter* painter,
	           const QStyleOptionGraphicsItem* option,
	           QWidget* widget = nullptr) override;
};
} // namespace __impl
} // namespace PVParallelView

#endif // __PVPARALLELVIEW_PVAXISHEADER_H__
