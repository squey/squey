
#ifndef PVPARALLELVIEW_PVAXISLABEL_H
#define PVPARALLELVIEW_PVAXISLABEL_H

#include <pvbase/types.h>

#include <QObject>
#include <QGraphicsSimpleTextItem>
#include <QGraphicsSceneContextMenuEvent>
#include <QBrush>

namespace Picviz
{

class PVView;

}

namespace PVParallelView
{

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

protected:
	virtual void contextMenuEvent(QGraphicsSceneContextMenuEvent *event);

signals:
	void new_zoomed_parallel_view(int _axis_index);

private slots:
	void new_zoomed_parallel_view();
	void new_selection_sliders();

private:
	const Picviz::PVView &_lib_view;
	PVSlidersGroup       *_sliders_group;
	PVCol                 _axis_index;
};

}

#endif // PVPARALLELVIEW_PVAXISLABEL_H
