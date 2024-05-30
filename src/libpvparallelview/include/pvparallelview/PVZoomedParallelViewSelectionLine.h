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

#ifndef PVPARALLELVIEW_PVZOOMEDPARALLELVIEWSELECTIONLINE_H
#define PVPARALLELVIEW_PVZOOMEDPARALLELVIEWSELECTIONLINE_H

#include <QGraphicsObject>

class QTimer;

namespace Squey
{

class PVView;
} // namespace Squey

namespace PVParallelView
{

class PVZoomedParallelView;

class PVZoomedParallelViewSelectionLine : public QGraphicsObject
{
	Q_OBJECT

  public:
	/**
	 * create a selection rectangle for hit-count view
	 *
	 * @param hcv the "parent" hit-count view
	 */
	explicit PVZoomedParallelViewSelectionLine(PVZoomedParallelView* zpv);

	/**
	 * DTOR!
	 */
	~PVZoomedParallelViewSelectionLine() override;

	/**
	 * return the bounding rectangle of the item
	 *
	 * @return the item's bounding rectangle
	 */
	QRectF boundingRect() const override;

	/**
	 * paint the item
	 *
	 * @param painter the painter to use
	 * @param option the QGrapghicsItem's style to use
	 * @param widget the widget requesting the paint (consider it as null)
	 */
	void paint(QPainter* painter,
	           const QStyleOptionGraphicsItem* option,
	           QWidget* widget = nullptr) override;

  public:
	/**
	 * clear the selection line
	 */
	void clear();

	/**
	 * test if the line is null of not
	 *
	 * @return true if the line is null; false otherwise
	 */
	bool is_null() const;

	/**
	 * get the top value (the one at the top of the view)
	 *
	 * @return the top value
	 */
	qreal top() const;

	/**
	 * get the bottom value (the one at the bottom of the view)
	 *
	 * @return the bottom value
	 */
	qreal bottom() const;

	/**
	 * start a selection line creation
	 *
	 * @param p the current scene coordinate
	 */
	void begin(const QPointF& p);

	/**
	 * process a step in selection line creation
	 *
	 * @param p the current scene coordinate
	 */
	void step(const QPointF& p, bool need_timer = true);

	/**
	 * finalize a selection line creation
	 *
	 * @param p the current scene coordinate
	 */
	void end(const QPointF& p);

  public:
	/**
	 * set the view related scale factor
	 *
	 * As the view can not be accessed, the transformation has to be
	 * passed using this method
	 *
	 * @param p the scene coordinate
	 */
	void set_view_scale(const qreal xscale, const qreal yscale);

  Q_SIGNALS:
	/**
	 * a signal emitted when the timeout occurs
	 */
	void commit_volatile_selection();

  protected:
	/**
	 * start the internal timer used to start the selection update
	 */
	void start_timer();

	/**
	 * retrieve the associated zoomed parallel view
	 *
	 * @return the zoomed parallel view
	 */
	PVZoomedParallelView* get_zpview() { return _zpv; }

  protected Q_SLOTS:
	/**
	 * the timeout slot
	 */
	void timeout();

  private:
	QTimer* _timer;
	PVZoomedParallelView* _zpv;
	QPointF _ref_pos;
	QPointF _tl_pos;
	QPointF _br_pos;
	QColor _pen_color;
	qreal _x_scale;
	qreal _y_scale;
};
} // namespace PVParallelView

#endif // PVZOOMEDPARALLELVIEWSELECTIONLINE_H
