#ifndef PVCUSTOMSTYLE_FILE_H
#define PVCUSTOMSTYLE_FILE_H

#include <QProxyStyle>

namespace PVInspector {

class PVCustomStyle: public QProxyStyle {
public:
	PVCustomStyle();

public:
	void drawComplexControl(ComplexControl control, const QStyleOptionComplex* option, QPainter* painter, const QWidget* widget = 0) const;
	void drawControl(ControlElement element, const QStyleOption * option, QPainter * painter, const QWidget * widget = 0) const;
};

}


#endif

