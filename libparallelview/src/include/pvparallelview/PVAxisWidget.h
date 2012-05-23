#ifndef PVPARALLELVIEW_PVAXISWIDGET_H
#define PVPARALLELVIEW_PVAXISWIDGET_H

#include <QGraphicsItem>

#include <picviz/PVAxis.h>

namespace PVParallelView
{

class PVAxisWidget : public QGraphicsItem
{
public:
	PVAxisWidget() {}
	PVAxisWidget(const Picviz::PVAxis &axis);
	PVAxisWidget(const QString &text);

	~PVAxisWidget() {}

	QRectF boundingRect () const;

	void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0);

private:
	void set_text(QString &name);

	void update_bbox();

private:
	QString _text;
	QRectF  _bbox;
};

}

#endif // PVPARALLELVIEW_PVAXISWIDGET_H
