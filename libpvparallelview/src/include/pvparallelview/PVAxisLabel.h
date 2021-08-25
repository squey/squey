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

#ifndef PVPARALLELVIEW_PVAXISLABEL_H
#define PVPARALLELVIEW_PVAXISLABEL_H

#include <QObject>
#include <QGraphicsSimpleTextItem>
#include <QGraphicsSceneContextMenuEvent>
#include <QBrush>
#include <QPainterPath>

namespace Inendi
{

class PVView;
} // namespace Inendi

namespace PVParallelView
{

class PVAxisGraphicsItem;

class PVAxisLabel : public QObject, public QGraphicsSimpleTextItem
{
	Q_OBJECT
  private:
	static constexpr int MAX_WIDTH =
	    120; /*!< The maximum width of a label in pixel. This value should be calculated later,
	           depend of the client's windows settings. */

  public:
	explicit PVAxisLabel(const Inendi::PVView& view, QGraphicsItem* parent = nullptr);

	~PVAxisLabel() override;

	/** Elide the text if it is longer than MAX_WIDTH.*/
	void set_text(const QString& text);

	void set_color(const QColor& color) { setBrush(color); }

	QRectF get_scene_bbox() { return mapRectToScene(boundingRect()); }

	void set_bounding_box_width(int width);
	bool contains(const QPointF& point) const override;
	QPainterPath shape() const override;
	QRectF boundingRect() const override;

  private:
	PVAxisGraphicsItem const* get_parent_axis() const;

  private:
	const Inendi::PVView& _lib_view;
	int _bounding_box_width = 0;
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVAXISLABEL_H
