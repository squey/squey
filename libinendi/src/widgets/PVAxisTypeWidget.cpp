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
	addItem("number_float");
	addItem("number_double");
	addItem("number_int32");
	addItem("number_uint32");
	addItem("time");
	addItem("ipv4");
	addItem("ipv6");
	addItem("mac_address");
}
