/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2019
 */

#include "PVPcapParamsWidget.h"
#include <libpvpcap/ws.h>

PVPcapsicum::PVPcapParamsWidget::PVPcapParamsWidget(QWidget* parent)
    : QDialog(parent), _selection_widget(new SelectionWidget)
{
	setLayout(new QVBoxLayout);
	layout()->addWidget(_selection_widget);
	resize(800, 500);

	connect(_selection_widget, &SelectionWidget::closed, this, &QDialog::accept);
	connect(_selection_widget, &SelectionWidget::canceled, this, &QDialog::reject);
}
