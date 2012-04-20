#ifndef PVSIZEHINTLISTWIDGET_H__
#define PVSIZEHINTLISTWIDGET_H__

#include <QtGui/QListWidget>

namespace PVWidgets {

class PVSizeHintListWidget : public QListWidget
{
public:
	PVSizeHintListWidget(QWidget * parent = 0) : QListWidget(parent) {}
	QSize sizeHint() const
	{
		return QSize(0, 40);
	}
};

}

#endif // PVSIZEHINTLISTWIDGET_H__
