#ifndef PVPARALLELVIEW_PVAXISWIDGETSELECTOR_H
#define PVPARALLELVIEW_PVAXISWIDGETSELECTOR_H

#include <QGraphicsItem>

namespace PVParallelView
{

class PVAxisWidgetSelector : public QGraphicsItem
{
public:
	PVAxisWidget() {}
	PVAxisWidget(uint32_t a, uint32_t b);
	PVAxisWidget(const QString &text);

	~PVAxisWidget() {}

	QRectF boundingRect () const;

	void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0);

private:
	void set_text(QString &name);

	void update_bbox();

private:
	uint32_t _a, _b;
};

}

#endif // PVPARALLELVIEW_PVAXISWIDGETSELECTOR_H
