//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include "PVFieldConverterGUIDToIPParamWidget.h"

#include <pvkernel/filter/PVFieldsFilter.h>
#include <QtCore/qobjectdefs.h>
#include <qstring.h>
#include <qtmetamacros.h>
#include <qvariant.h>
#include <qwidget.h>
#include <QAction>
#include <QRadioButton>
#include <QGroupBox>
#include <QLabel>
#include <QVBoxLayout>
#include <memory>

#include "PVFieldGUIDToIP.h"
#include "pvkernel/core/PVArgument.h"
#include "pvkernel/core/PVLogger.h"
#include "pvkernel/core/PVOrderedMap.h"
#include "pvkernel/filter/PVFieldsFilterParamWidget.h"

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

	auto* layout = new QVBoxLayout(_param_widget);

	_param_widget->setLayout(layout);
	_param_widget->setObjectName("splitter");

	// Label
	auto* label = new QLabel("This converter translates GUID into IPv4 or IPv6 address.\n"
	                           "\n"
	                           "GUID={58BEFDF8-400C-0C00-0000-000000000071}\n"
	                           "IPv4=88.190.253.248\n"
	                           "IPv6=58be:fdf8:400c:0c00:0000:0000:0000:0071\n"
	                           "\n"
	                           "Note that potential enclosing characters such as \n"
	                           "{} will be  automatically discarded.\n");
	layout->addWidget(label);

	// Checkboxes
	auto* groupBox = new QGroupBox("IP type");
	auto* ips_layout = new QVBoxLayout();
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
