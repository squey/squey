/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVFieldConverterGUIDToIPParamWidget.h"
#include "PVFieldGUIDToIP.h"

#include <pvkernel/filter/PVElementFilterByFields.h>
#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/widgets/qkeysequencewidget.h>

#include <QAction>
#include <QRadioButton>
#include <QGroupBox>
#include <QLabel>
#include <QVBoxLayout>

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterGUIDToIPParamWidget::PVFieldSplitterCSVParamWidget
 *
 *****************************************************************************/
PVFilter::PVFieldConverterGUIDToIPParamWidget::PVFieldConverterGUIDToIPParamWidget()
    : PVFieldsConverterParamWidget(PVFilter::PVFieldsConverter_p(new PVFieldGUIDToIP()))
{
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterGUIDToIPParamWidget::get_action_menu
 *
 *****************************************************************************/
QAction* PVFilter::PVFieldConverterGUIDToIPParamWidget::get_action_menu(QWidget* parent)
{
	return new QAction(QString("add GUID to IP Converter"), parent);
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterGUIDToIPParamWidget::get_param_widget
 *
 *****************************************************************************/
QWidget* PVFilter::PVFieldConverterGUIDToIPParamWidget::get_param_widget()
{
	PVLOG_DEBUG("PVFilter::PVFieldConverterGUIDToIPParamWidget::get_param_widget()     start\n");

	PVCore::PVArgumentList args = get_filter()->get_args();

	_param_widget = new QWidget();

	QVBoxLayout* layout = new QVBoxLayout(_param_widget);

	_param_widget->setLayout(layout);
	_param_widget->setObjectName("splitter");

	// Label
	QLabel* label = new QLabel("This converter translates GUID into IPv4 or IPv6 address.\n"
	                           "\n"
	                           "GUID={58BEFDF8-400C-0C00-0000-000000000071}\n"
	                           "IPv4=88.190.253.248\n"
	                           "IPv6=58be:fdf8:400c:0c00:0000:0000:0000:0071\n"
	                           "\n"
	                           "Note that potential enclosing characters such as \n"
	                           "{} will be  automatically discarded.\n");
	layout->addWidget(label);

	// Checkboxes
	QGroupBox* groupBox = new QGroupBox("IP type");
	QVBoxLayout* ips_layout = new QVBoxLayout();
	_ipv4 = new QRadioButton(tr("IPv4"));
	_ipv4->setChecked(!args["ipv6"].toBool());
	_ipv6 = new QRadioButton(tr("IPv6"));
	_ipv6->setChecked(args["ipv6"].toBool());
	ips_layout->addWidget(_ipv4);
	ips_layout->addWidget(_ipv6);
	groupBox->setLayout(ips_layout);
	layout->addWidget(groupBox);
	connect(_ipv6, &QRadioButton::toggled, [this](bool) { update_params(); });

	return _param_widget;
}

void PVFilter::PVFieldConverterGUIDToIPParamWidget::update_params()
{
	PVCore::PVArgumentList args = get_filter()->get_args();

	args["ipv6"] = _ipv6->isChecked();

	get_filter()->set_args(args);
	Q_EMIT args_changed_Signal();
}
