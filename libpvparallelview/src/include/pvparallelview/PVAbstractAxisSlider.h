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

#ifndef PVPARALLELVIEW_PVABSTRACTAXISSLIDER_H
#define PVPARALLELVIEW_PVABSTRACTAXISSLIDER_H

#include <pvkernel/core/PVAlgorithms.h>

#include <QObject>
#include <QGraphicsItem>
#include <QGraphicsSceneContextMenuEvent>

/* TODO: move from int to uint32_t to use QPointF instead of QPoint to  move precisely
 *       any sliders in zoomed view
 */

namespace PVParallelView
{

class PVAbstractAxisSliders;

enum PVAxisSliderOrientation { Min, Max };

class PVAbstractAxisSlider : public QGraphicsObject
{
	Q_OBJECT

  public:
	constexpr static int64_t min_value = 0LL;
	constexpr static int64_t max_value = (1LL << 32);

  public:
	PVAbstractAxisSlider(int64_t omin,
	                     int64_t omax,
	                     int64_t o,
	                     PVAxisSliderOrientation orientation = Min,
	                     QGraphicsItem* parent_item = nullptr);

	~PVAbstractAxisSlider() override;

	void set_value(int64_t v);

	inline int64_t get_value() const { return _offset; }

	void set_range(int64_t omin, int64_t omax)
	{
		_offset_min = omin;
		_offset_max = omax;
	}

	void set_owner(PVAbstractAxisSliders* owner) { _owner = owner; }

	bool is_moving() const { return _moving; }

  public:
	void paint(QPainter* painter,
	           const QStyleOptionGraphicsItem* option,
	           QWidget* widget = nullptr) override;

  Q_SIGNALS:
	void slider_moved();

  protected:
	void hoverEnterEvent(QGraphicsSceneHoverEvent* event) override;
	void hoverMoveEvent(QGraphicsSceneHoverEvent* event) override;
	void hoverLeaveEvent(QGraphicsSceneHoverEvent* event) override;

	void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
	void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
	void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;

	void contextMenuEvent(QGraphicsSceneContextMenuEvent* event) override;

	bool mouse_is_hover() const { return _is_hover; }

  protected:
	int64_t _offset_min;
	int64_t _offset_max;
	int64_t _offset;
	double _move_offset;
	PVAxisSliderOrientation _orientation;
	bool _moving;
	bool _is_hover;
	PVAbstractAxisSliders* _owner;
	bool _removable;
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVABSTRACTAXISSLIDER_H
