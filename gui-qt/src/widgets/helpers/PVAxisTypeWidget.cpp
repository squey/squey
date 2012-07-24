/**
 * \file PVAxisTypeWidget.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <PVAxisTypeWidget.h>
#include <picviz/PVMappingFilter.h>

PVInspector::PVWidgetsHelpers::PVAxisTypeWidget::PVAxisTypeWidget(QWidget* parent):
	PVComboBox(parent)
{
	addItems(Picviz::PVMappingFilter::list_types());
}
