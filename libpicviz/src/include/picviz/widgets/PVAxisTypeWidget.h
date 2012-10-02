/**
 * \file PVAxisTypeWidget.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVWIDGETS_PVAXISTYPEWIDGET_H
#define PVWIDGETS_PVAXISTYPEWIDGET_H

#include <pvkernel/widgets/PVComboBox.h>

namespace PVWidgets{

class PVAxisTypeWidget: public PVComboBox
{
public:
	PVAxisTypeWidget(QWidget* parent = NULL);
public:
	inline QString get_sel_type() const { return currentText(); }
	inline bool sel_type(QString const& type) { return select(type); }
};


}

#endif
