/**
 * \file PVSizeHintListWidget.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVSIZEHINTLISTWIDGET_H__
#define PVSIZEHINTLISTWIDGET_H__

#include <QtGui/QListWidget>

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
