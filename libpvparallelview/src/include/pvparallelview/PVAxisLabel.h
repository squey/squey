
#ifndef PVPARALLELVIEW_PVAXISLABEL_H
#define PVPARALLELVIEW_PVAXISLABEL_H

#include <pvbase/types.h>

#include <QObject>
#include <QGraphicsSimpleTextItem>
#include <QGraphicsSceneContextMenuEvent>
#include <QBrush>
#include <QPainterPath>

namespace Picviz
{

class PVView;

}

namespace PVParallelView
{

class PVAxisGraphicsItem;

class PVSlidersGroup;

class PVAxisLabel : public QObject, public QGraphicsSimpleTextItem
{
Q_OBJECT

public:
	PVAxisLabel(const Picviz::PVView &view, PVSlidersGroup *sg,
	            QGraphicsItem *parent = nullptr);

	virtual ~PVAxisLabel();

	void set_text(const QString &text)
	{
		setText(text);
	}

	void set_color(const QColor &color)
	{
		setBrush(color);
	}

	void set_axis_index(const PVCol index)
	{
		_axis_index = index;
	}

	QRectF get_scene_bbox()
	{
		return mapRectToScene(boundingRect());
	}

	void set_bounding_box_width(int width);
	bool contains(const QPointF & point) const override;
	QPainterPath shape() const override;
	QRectF boundingRect() const;

	PVCol get_axis_index() const;

private:
	PVAxisGraphicsItem const* get_parent_axis() const;

private:
	const Picviz::PVView &_lib_view;
	PVSlidersGroup       *_sliders_group;
	PVCol                 _axis_index;
	int 				  _bounding_box_width = 0;
};

}

#endif // PVPARALLELVIEW_PVAXISLABEL_H
