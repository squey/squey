#ifndef PVSIZEHINTLISTWIDGET_H__
#define PVSIZEHINTLISTWIDGET_H__

#include <QtGui/QListWidget>

namespace PVWidgets {

template <class T = QListWidget>
class PVSizeHintListWidget : public T
{
public:
	PVSizeHintListWidget(QWidget * parent = 0) : T(parent) {}
	QSize sizeHint() const
	{
		return QSize(0, 40);
	}
};

}

#endif // PVSIZEHINTLISTWIDGET_H__
