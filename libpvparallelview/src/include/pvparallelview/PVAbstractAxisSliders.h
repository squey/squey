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

#ifndef PVPARALLELVIEW_PVABSTRACTAXISSLIDERS_H
#define PVPARALLELVIEW_PVABSTRACTAXISSLIDERS_H

#include <pvparallelview/PVSlidersManager.h>

#include <QObject>
#include <QGraphicsItemGroup>

class QGraphicsSimpleTextItem;

namespace PVParallelView
{

class PVSlidersGroup;

class PVAbstractAxisSliders : public QObject, public QGraphicsItemGroup
{
	Q_OBJECT

  public:
	PVAbstractAxisSliders(QGraphicsItem* parent,
	                      PVSlidersManager* sm_p,
	                      PVSlidersGroup* group,
	                      const char* text);

	inline PVSlidersGroup* get_sliders_group() const { return _group; }

	virtual bool is_moving() const = 0;

	virtual void refresh() = 0;

	QRectF boundingRect() const override;
	void paint(QPainter* painter,
	           const QStyleOptionGraphicsItem* option,
	           QWidget* widget = nullptr) override;

  public Q_SLOTS:
	virtual void remove_from_axis() = 0;

  Q_SIGNALS:
	void sliders_moved();

  protected:
	PVSlidersManager* _sliders_manager_p;
	PVSlidersGroup* _group;
	QGraphicsSimpleTextItem* _text;
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVABSTRACTAXISSLIDERS_H
