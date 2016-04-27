/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include <inendi/PVMappingFilter.h>
#include <inendi/widgets/PVAxisTypeWidget.h>

PVWidgets::PVAxisTypeWidget::PVAxisTypeWidget(QString const& current_type, QWidget* parent)
    : PVComboBox(parent)
{
	if (current_type == "all") {
		addItems(Inendi::PVMappingFilter::list_types());
		return;
	} else if (current_type == "string" or current_type == "host" or current_type == "enum") {
		addItem("string");
		addItem("host");
		addItem("enum");
	} else if (current_type == "float") {
		addItem("float");
	} else if (current_type == "integer") {
		addItem("integer");
	} else if (current_type == "time") {
		addItem("time");
	} else if (current_type == "ipv4") {
		addItem("ipv4");
	}
	select(current_type);
}
