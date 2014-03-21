/**
 * \file PVFieldSplitterDnsFqdnParamWidget.cpp
 *
 * Copyright (C) Picviz Labs 2014
 */

#include "PVFieldSplitterDnsFqdnParamWidget.h"
#include "PVFieldSplitterDnsFqdn.h"
#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/filter/PVElementFilterByFields.h>

#include <QLabel>
#include <QGridLayout>

/******************************************************************************
 * PVFilter::PVFieldSplitterDnsFqdnParamWidget::PVFieldSplitterCSVParamWidget
 *****************************************************************************/

PVFilter::PVFieldSplitterDnsFqdnParamWidget::PVFieldSplitterDnsFqdnParamWidget() :
	PVFieldsSplitterParamWidget(PVFilter::PVFieldsSplitter_p(new PVFieldSplitterDnsFqdn()))
{
	_action_menu = new QAction(QString("add DNS FQDN Splitter"), NULL);
}

/******************************************************************************
 * PVFilter::PVFieldSplitterDnsFqdnParamWidget::get_action_menu
 *****************************************************************************/

QAction* PVFilter::PVFieldSplitterDnsFqdnParamWidget::get_action_menu()
{
    assert(_action_menu);
    return _action_menu;
}

/******************************************************************************
 * PVFilter::PVFieldSplitterDnsFqdnParamWidget::get_param_widget
 *****************************************************************************/

QWidget* PVFilter::PVFieldSplitterDnsFqdnParamWidget::get_param_widget()
{
	//get args
	PVCore::PVArgumentList args =  get_filter()->get_args();

	_n             = args[PVFieldSplitterDnsFqdn::N].toInt();
	bool tld1      = args[PVFieldSplitterDnsFqdn::TLD1].toBool();
	bool tld2      = args[PVFieldSplitterDnsFqdn::TLD2].toBool();
	bool tld3      = args[PVFieldSplitterDnsFqdn::TLD3].toBool();
	bool subd1     = args[PVFieldSplitterDnsFqdn::SUBD1].toBool();
	bool subd2     = args[PVFieldSplitterDnsFqdn::SUBD2].toBool();
	bool subd3     = args[PVFieldSplitterDnsFqdn::SUBD3].toBool();
	bool subd1_rev = args[PVFieldSplitterDnsFqdn::SUBD1_REV].toBool();
	bool subd2_rev = args[PVFieldSplitterDnsFqdn::SUBD2_REV].toBool();
	bool subd3_rev = args[PVFieldSplitterDnsFqdn::SUBD3_REV].toBool();

	set_child_count(_n);
	emit nchilds_changed_Signal();

	_param_widget = new QWidget();

	QGridLayout* layout = new QGridLayout(_param_widget);

	layout->setSizeConstraint(QLayout::SetFixedSize);

	_param_widget->setLayout(layout);
	_param_widget->setObjectName("splitter");

	_split_cb[0] = new QCheckBox("extract TLD1");
	_split_cb[0]->setToolTip("get the first top level domain");
	_split_cb[0]->setChecked(tld1);

	_split_cb[1] = new QCheckBox("extract TLD2");
	_split_cb[1]->setToolTip("get the two first top level domains");
	_split_cb[1]->setChecked(tld2);

	_split_cb[2] = new QCheckBox("extract TLD3");
	_split_cb[2]->setToolTip("get the three first top level domains");
	_split_cb[2]->setChecked(tld3);

	_split_cb[3] = new QCheckBox("extract SUBD1");
	_split_cb[3]->setToolTip("get all but the first top level domain");
	_split_cb[3]->setChecked(subd1);

	_split_cb[4] = new QCheckBox("extract SUBD2");
	_split_cb[4]->setToolTip("get all but the two first top level domains");
	_split_cb[4]->setChecked(subd2);

	_split_cb[5] = new QCheckBox("extract SUBD3");
	_split_cb[5]->setToolTip("get all but the three first top level domains");
	_split_cb[5]->setChecked(subd3);

	for (int i = 0; i < 6; ++i) {
		QWidget *w = _split_cb[i];
		connect(w, SIGNAL(stateChanged(int)), this, SLOT(split_cb_changed(int)));
		layout->addWidget(w, i, 0);
	}

	_rev_cb[0] = new QCheckBox("invert IP if a PTR query");
	_rev_cb[0]->setChecked(subd1_rev);
	_rev_cb[1] = new QCheckBox("invert IP if a PTR query");
	_rev_cb[1]->setChecked(subd2_rev);
	_rev_cb[2] = new QCheckBox("invert IP if a PTR query");
	_rev_cb[2]->setChecked(subd3_rev);

	for (int i = 0; i < 3; ++i) {
		QWidget *w = _rev_cb[i];
		w->setEnabled(_split_cb[i + 3]->isChecked());
		connect(w, SIGNAL(stateChanged(int)), this, SLOT(rev_cb_changed(int)));
		layout->addWidget(w, i + 3, 1);
	}

	return _param_widget;
}

/******************************************************************************
 * PVFilter::PVFieldSplitterDnsFqdnParamWidget::split_cb_changed
 *****************************************************************************/

void PVFilter::PVFieldSplitterDnsFqdnParamWidget::split_cb_changed(int)
{
	PVCore::PVArgumentList args = get_filter()->get_args();

	_n = 0;

	for(int i = 0; i < 6; ++i) {
		bool s = _split_cb[i]->isChecked();
		if (s) {
			++_n;
		}
		if (i >= 3) {
			// need to enable/disable subdomains rev cb
			_rev_cb[i-3]->setEnabled(s);
		}
	}

	update_args(args);

	get_filter()->set_args(args);

	set_child_count(_n);

	emit nchilds_changed_Signal();
	emit args_changed_Signal();
}

/******************************************************************************
 * PVFilter::PVFieldSplitterDnsFqdnParamWidget::rev_cb_changed
 *****************************************************************************/

void PVFilter::PVFieldSplitterDnsFqdnParamWidget::rev_cb_changed(int)
{
	PVCore::PVArgumentList args = get_filter()->get_args();

	update_args(args);

	get_filter()->set_args(args);

	emit args_changed_Signal();
}

/******************************************************************************
 * PVFilter::PVFieldSplitterDnsFqdnParamWidget::update_child_count
 *****************************************************************************/

void PVFilter::PVFieldSplitterDnsFqdnParamWidget::update_args(PVCore::PVArgumentList& args)
{
	args[PVFieldSplitterDnsFqdn::N] = _n;

	args[PVFieldSplitterDnsFqdn::TLD1] = _split_cb[0]->isChecked();
	args[PVFieldSplitterDnsFqdn::TLD2] = _split_cb[1]->isChecked();
	args[PVFieldSplitterDnsFqdn::TLD3] = _split_cb[2]->isChecked();

	args[PVFieldSplitterDnsFqdn::SUBD1] = _split_cb[3]->isChecked();
	args[PVFieldSplitterDnsFqdn::SUBD2] = _split_cb[4]->isChecked();
	args[PVFieldSplitterDnsFqdn::SUBD3] = _split_cb[5]->isChecked();

	args[PVFieldSplitterDnsFqdn::SUBD1_REV] = _rev_cb[0]->isChecked();
	args[PVFieldSplitterDnsFqdn::SUBD2_REV] = _rev_cb[1]->isChecked();
	args[PVFieldSplitterDnsFqdn::SUBD3_REV] = _rev_cb[2]->isChecked();
}
