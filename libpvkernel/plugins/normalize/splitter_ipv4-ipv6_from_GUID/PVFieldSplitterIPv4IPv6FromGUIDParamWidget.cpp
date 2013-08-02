/**
 * \file PVFieldSplitterIPv4IPv6FromGUIDParamWidget.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

#include "PVFieldSplitterIPv4IPv6FromGUIDParamWidget.h"
#include "PVFieldIPv4IPv6FromGUID.h"
#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/filter/PVElementFilterByFields.h>

#include <QAction>
#include <QCheckBox>
#include <QComboBox>
#include <QGroupBox>
#include <QLabel>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QSpacerItem>
#include <QMessageBox>

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterDuplicateParamWidget::PVFieldSplitterCSVParamWidget
 *
 *****************************************************************************/
PVFilter::PVFieldSplitterIPv4IPv6FromGUIDParamWidget::PVFieldSplitterIPv4IPv6FromGUIDParamWidget() :
	PVFieldsSplitterParamWidget(PVFilter::PVFieldsSplitter_p(new PVFieldIPv4IPv6FromGUID()))
{
	_action_menu = new QAction(QString("add IPv4-IPv6 From GUID Splitter"), NULL);
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterDuplicateParamWidget::get_action_menu
 *
 *****************************************************************************/
QAction* PVFilter::PVFieldSplitterIPv4IPv6FromGUIDParamWidget::get_action_menu()
{
    PVLOG_DEBUG("get action PVFieldSplitterIPv4IPv6FromGUIDParamWidget\n");
    assert(_action_menu);
    return _action_menu;
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterDuplicateParamWidget::get_param_widget
 *
 *****************************************************************************/
QWidget* PVFilter::PVFieldSplitterIPv4IPv6FromGUIDParamWidget::get_param_widget()
{
	PVLOG_DEBUG("PVFilter::PVFieldSplitterIPv4IPv6FromGUIDParamWidget::get_param_widget()     start\n");

	PVCore::PVArgumentList args = get_filter()->get_args();

	_param_widget = new QWidget();

	QVBoxLayout* layout = new QVBoxLayout(_param_widget);

	_param_widget->setLayout(layout);
	_param_widget->setObjectName("splitter");

	// Label
	QLabel* label = new QLabel(
		"This splitter converts GUID into IPv4 and/or IPv6.\n"
		"\n"
		"GUID={7F000001-FFFF-0000-0000-000000000000}\n"
		"IPv4=127.0.0.1\n"
		"IPv6=FFFF-0000-0000-000000000000\n"
		"\n"
		"Note that potential enclosing characters such as \n"
		"{} will be discarded.\n"
	);
	layout->addWidget(label);

	// Checkboxes
	QGroupBox* groupBox = new QGroupBox("IP addresses");
	QVBoxLayout* ips_layout = new QVBoxLayout();
	_ipv4_cb = new QCheckBox(tr("IPv4"));
	_ipv4_cb->setChecked(args["ipv4"].toBool());
	_ipv6_cb = new QCheckBox(tr("IPv6"));
	_ipv6_cb->setChecked(args["ipv6"].toBool());
	ips_layout->addWidget(_ipv4_cb);
	ips_layout->addWidget(_ipv6_cb);
	groupBox->setLayout(ips_layout);
	layout->addWidget(groupBox);
	connect(_ipv4_cb, SIGNAL(stateChanged(int)), this, SLOT(updateNChilds()));
	connect(_ipv6_cb, SIGNAL(stateChanged(int)), this, SLOT(updateNChilds()));

	return _param_widget;
}


void PVFilter::PVFieldSplitterIPv4IPv6FromGUIDParamWidget::updateNChilds()
{
	PVCore::PVArgumentList args = get_filter()->get_args();

	size_t n = (size_t)_ipv4_cb->isChecked() + (size_t)_ipv6_cb->isChecked();

	if (n == 0) {
		if (sender() == _ipv4_cb) {
			_ipv4_cb->setCheckState(Qt::Checked);
		}
		else if (sender() == _ipv6_cb) {
			_ipv6_cb->setCheckState(Qt::Checked);
		}
		QMessageBox box(QMessageBox::Critical, tr("No IP type selected"), tr("At least one IP type must be selected."));
		box.exec();
	}

	args["ipv4"] = _ipv4_cb->isChecked();
	args["ipv6"] = _ipv6_cb->isChecked();

	n = (size_t)_ipv4_cb->isChecked() + (size_t)_ipv6_cb->isChecked();

	get_filter()->set_args(args);
	set_child_count(n);
    emit args_changed_Signal();
	emit nchilds_changed_Signal();
}
