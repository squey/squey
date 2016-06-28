/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include <inendi/widgets/PVAxisTypeWidget.h>

PVWidgets::PVAxisTypeWidget::PVAxisTypeWidget(QWidget* parent) : PVComboBox(parent)
{
	addItem("string");
	addItem("float");
	addItem("integer");
	addItem("time");
	addItem("ipv4");
}
