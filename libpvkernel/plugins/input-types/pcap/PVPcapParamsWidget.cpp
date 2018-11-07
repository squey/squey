/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2017
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
