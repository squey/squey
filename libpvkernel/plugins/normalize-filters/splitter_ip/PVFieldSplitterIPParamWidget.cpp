/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVFieldSplitterIPParamWidget.h"
#include "PVFieldSplitterIP.h"
#include <pvkernel/filter/PVFieldsFilter.h>

#include <QAction>
#include <QRadioButton>
#include <QGroupBox>
#include <QCheckBox>
#include <QLabel>
#include <QHBoxLayout>
#include <QSpacerItem>

constexpr size_t group_ipv4_count = 4;
constexpr size_t group_ipv6_count = 8;

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterIPParamWidget::PVFieldSplitterCSVParamWidget
 *
 *****************************************************************************/
PVFilter::PVFieldSplitterIPParamWidget::PVFieldSplitterIPParamWidget()
    : PVFieldsSplitterParamWidget(PVFilter::PVFieldsSplitter_p(new PVFieldSplitterIP()))
{
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterIPParamWidget::get_action_menu
 *
 *****************************************************************************/
QAction* PVFilter::PVFieldSplitterIPParamWidget::get_action_menu(QWidget* parent)
{
	return new QAction(QString("add IP Splitter"), parent);
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterIPParamWidget::get_param_widget
 *
 *****************************************************************************/

QWidget* PVFilter::PVFieldSplitterIPParamWidget::get_param_widget()
{
	PVLOG_DEBUG("PVFilter::PVFieldSplitterIPParamWidget::get_param_widget()\n");

	QWidget* param_widget = new QWidget();

	// get args
	PVCore::PVArgumentList args = get_filter()->get_args();

	QVBoxLayout* layout = new QVBoxLayout(param_widget);

	param_widget->setLayout(layout);
	param_widget->setObjectName("splitter");

	QGroupBox* groupBox = new QGroupBox("IP type");
	QVBoxLayout* ips_layout = new QVBoxLayout();
	_ipv4 = new QRadioButton(tr("IPv4"));
	_ipv6 = new QRadioButton(tr("IPv6"));
	ips_layout->addWidget(_ipv4);
	ips_layout->addWidget(_ipv6);
	groupBox->setLayout(ips_layout);
	layout->addWidget(groupBox);
	_ipv4->setChecked(!args["ipv6"].toBool());
	_ipv6->setChecked(args["ipv6"].toBool());
	connect(_ipv4, &QAbstractButton::toggled, this, &PVFieldSplitterIPParamWidget::set_ip_type);

	_label_list.clear();
	_cb_list.clear();
	QVBoxLayout* groups_layout = new QVBoxLayout();
	for (size_t i = 0; i < group_ipv6_count; i++) {
		QLabel* label = new QLabel(QString("Group%1").arg(i + 1));
		_label_list.append(label);
		groups_layout->addWidget(label, 0, Qt::AlignHCenter);
		if (i < group_ipv6_count - 1) {
			QCheckBox* checkbox = new QCheckBox();
			checkbox->setStyleSheet(
			    "QCheckBox::indicator {width: 22px; height: 22px; }"
			    "QCheckBox::indicator:checked{ image: url(:/scissors_on); }"
			    "QCheckBox::indicator:unchecked{ image: url(:/scissors_off); }");
			checkbox->setCursor(QCursor(Qt::PointingHandCursor));
			connect(checkbox, &QCheckBox::stateChanged, this,
			        &PVFieldSplitterIPParamWidget::update_child_count);
			_cb_list.append(checkbox);
			groups_layout->addWidget(_cb_list[i], 0, Qt::AlignHCenter);
		}
	}
	groups_layout->setAlignment(Qt::AlignHCenter);
	QWidget* groups_widget = new QWidget();
	groups_widget->setLayout(groups_layout);
	layout->addWidget(groups_widget);

	set_ip_type(/*reset_groups_check_state=*/false);

	set_groups_check_state();

	return param_widget;
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterIPParamWidget::set_ip_type
 *
 *****************************************************************************/
void PVFilter::PVFieldSplitterIPParamWidget::set_ip_type(bool reset_groups_check_state /*= true*/)
{
	PVLOG_DEBUG("PVFilter::PVFieldSplitterIPParamWidget::update_ip_type\n");

	PVCore::PVArgumentList args = get_filter()->get_args();
	args["ipv6"] = _ipv6->isChecked();
	_group_count = _ipv6->isChecked() ? group_ipv6_count : group_ipv4_count;

	// resetting 'params'
	if (_ipv6->isChecked()) {
		args["params"] = PVFieldSplitterIP::params_ipv6;
	} else {
		args["params"] = PVFieldSplitterIP::params_ipv4;
	}

	get_filter()->set_args(args);
	Q_EMIT args_changed_Signal();

	if (reset_groups_check_state) {
		set_groups_check_state(/*check_all=*/true);
	}

	bool ipv6 = _ipv6->isChecked();
	if (ipv6) {
		for (size_t i = 0; i < group_ipv6_count; i++) {
			_label_list[i]->setText(QString("Group%1").arg(i));
		}
	} else {
		for (size_t i = 0; i < group_ipv4_count; i++) {
			_label_list[i]->setText(QString("Quad%1").arg(i));
		}
	}
	for (size_t i = group_ipv4_count - 1; i < group_ipv6_count; i++) {
		if (i < group_ipv6_count - 1) {
			_cb_list[i]->setVisible(ipv6);
		}
		if (i > group_ipv4_count - 1) {
			_label_list[i]->setVisible(ipv6);
		}
	}
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterIPParamWidget::set_groups_check_state
 *
 *****************************************************************************/
void PVFilter::PVFieldSplitterIPParamWidget::set_groups_check_state(bool check_all /* = false */)
{
	PVCore::PVArgumentList args = get_filter()->get_args();

	QString p = args["params"].toString();
	QStringList pl = p.split(PVFieldSplitterIP::sep);

	if (check_all) {
		bool ipv6 = _ipv6->isChecked();
		QStringList list;
		if (ipv6) {
			for (size_t i = 0; i < group_ipv6_count - 1; i++) {
				_cb_list[i]->setChecked(true);
				list << QString::number(i);
			}
		} else {
			for (size_t i = 0; i < group_ipv4_count - 1; i++) {
				_cb_list[i]->setChecked(true);
				list << QString::number(i);
			}
		}
		args["params"] = list.join(PVFieldSplitterIP::sep);
		get_filter()->set_args(args);
		set_child_count(list.size() + 1);
		Q_EMIT args_changed_Signal();
		Q_EMIT nchilds_changed_Signal();
	} else {
		for (const QString& s : pl) {
			size_t index = s.toUInt();
			if (index < group_ipv6_count) {
				_cb_list[index]->setChecked(true);
			}
		}
	}
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterIPParamWidget::update_child_count
 *
 *****************************************************************************/
void PVFilter::PVFieldSplitterIPParamWidget::update_child_count()
{
	PVCore::PVArgumentList args = get_filter()->get_args();

	size_t n = 1;

	QStringList list;
	for (size_t i = 0; i < _group_count - 1; i++) {
		if (_cb_list[i]->isChecked()) {
			list << QString::number(i);
			n++;
		}
	}
	args["params"] = list.join(PVFieldSplitterIP::sep);

	get_filter()->set_args(args);
	set_child_count(n);
	Q_EMIT args_changed_Signal();
	Q_EMIT nchilds_changed_Signal();
}
