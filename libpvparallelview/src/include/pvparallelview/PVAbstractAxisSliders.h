/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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
