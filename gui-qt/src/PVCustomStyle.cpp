/**
 * \file PVCustomStyle.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <PVCustomStyle.h>
#include <QPlastiqueStyle>
#include <QWindowsStyle>
#include <QCleanlooksStyle>

PVInspector::PVCustomStyle::PVCustomStyle():
	QProxyStyle(/*new QGtkStyle()*/)
{
}

void PVInspector::PVCustomStyle::drawComplexControl(ComplexControl control, const QStyleOptionComplex* option, QPainter* painter, const QWidget* widget) const
{
	switch (control) {
		//case QStyle::CC_TitleBar:
		//	return;
		default:
			QProxyStyle::drawComplexControl(control, option, painter, widget);
	};
}

void PVInspector::PVCustomStyle::drawControl(ControlElement element, const QStyleOption * option, QPainter * painter, const QWidget * widget) const
{
	switch (element) {
		//case QStyle::CE_PushButton:
		//	break;
		default:
			QProxyStyle::drawControl(element, option, painter, widget);
	};
}
