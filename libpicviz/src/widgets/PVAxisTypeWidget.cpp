/**
 * \file PVAxisTypeWidget.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <picviz/PVMappingFilter.h>
#include <picviz/widgets/PVAxisTypeWidget.h>

PVWidgets::PVAxisTypeWidget::PVAxisTypeWidget(QWidget* parent):
	PVComboBox(parent)
{
	addItems(Picviz::PVMappingFilter::list_types());
}
