/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include <inendi/widgets/PVAxisTypeWidget.h>

PVWidgets::PVAxisTypeWidget::PVAxisTypeWidget(QString const& current_type, QWidget* parent)
    : PVComboBox(parent)
{
	if (current_type == "all") {
		addItem("string");
		addItem("float");
		addItem("integer");
		addItem("time");
		addItem("ipv4");
	} else {
		addItem(current_type);
		select(current_type);
	}
}
