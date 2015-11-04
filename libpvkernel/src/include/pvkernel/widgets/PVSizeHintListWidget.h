/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVSIZEHINTLISTWIDGET_H__
#define PVSIZEHINTLISTWIDGET_H__

#include <QListWidget>

namespace PVWidgets {

template <class T = QListWidget, int VSize=42>
class PVSizeHintListWidget : public T
{
public:
	PVSizeHintListWidget(QWidget * parent = 0) : T(parent) {}
	QSize sizeHint() const
	{
		return QSize(0, VSize);
	}
};

}

#endif // PVSIZEHINTLISTWIDGET_H__
