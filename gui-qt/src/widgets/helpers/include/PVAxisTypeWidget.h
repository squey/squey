/**
 * \file PVAxisTypeWidget.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef WIDGETS_PVAXISTYPEWIDGET_H
#define WIDGETS_PVAXISTYPEWIDGET_H

#include <PVComboBox.h>

namespace PVInspector {

namespace PVWidgetsHelpers {

class PVAxisTypeWidget: public PVComboBox
{
public:
	PVAxisTypeWidget(QWidget* parent = NULL);
public:
	inline QString get_sel_type() const { return currentText(); }
	inline bool sel_type(QString const& type) { return select(type); }
};

}

}

#endif
