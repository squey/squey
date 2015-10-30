/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <picviz/PVMappingFilter.h>
#include <picviz/widgets/PVAxisTypeWidget.h>

PVWidgets::PVAxisTypeWidget::PVAxisTypeWidget(QWidget* parent):
	PVComboBox(parent)
{
	addItems(Picviz::PVMappingFilter::list_types());
}
