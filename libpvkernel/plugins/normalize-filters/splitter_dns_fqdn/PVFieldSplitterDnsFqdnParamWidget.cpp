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

#include "PVFieldSplitterDnsFqdnParamWidget.h"
#include "PVFieldSplitterDnsFqdn.h"
#include <pvkernel/filter/PVFieldsFilter.h>

#include <QLabel>
#include <QGridLayout>

/******************************************************************************
 * PVFilter::PVFieldSplitterDnsFqdnParamWidget::PVFieldSplitterCSVParamWidget
 *****************************************************************************/

PVFilter::PVFieldSplitterDnsFqdnParamWidget::PVFieldSplitterDnsFqdnParamWidget()
    : PVFieldsSplitterParamWidget(PVFilter::PVFieldsSplitter_p(new PVFieldSplitterDnsFqdn()))
{
}

/******************************************************************************
 * PVFilter::PVFieldSplitterDnsFqdnParamWidget::get_action_menu
 *****************************************************************************/

QAction* PVFilter::PVFieldSplitterDnsFqdnParamWidget::get_action_menu(QWidget* parent)
{
	return new QAction(QString("add DNS FQDN Splitter"), parent);
}

/******************************************************************************
 * PVFilter::PVFieldSplitterDnsFqdnParamWidget::get_param_widget
 *****************************************************************************/

QWidget* PVFilter::PVFieldSplitterDnsFqdnParamWidget::get_param_widget()
{
	// get args
	PVCore::PVArgumentList args = get_filter()->get_args();

	bool tld1 = args[PVFieldSplitterDnsFqdn::TLD1].toBool();
	bool tld2 = args[PVFieldSplitterDnsFqdn::TLD2].toBool();
	bool tld3 = args[PVFieldSplitterDnsFqdn::TLD3].toBool();
	bool subd1 = args[PVFieldSplitterDnsFqdn::SUBD1].toBool();
	bool subd2 = args[PVFieldSplitterDnsFqdn::SUBD2].toBool();
	bool subd3 = args[PVFieldSplitterDnsFqdn::SUBD3].toBool();
	bool subd1_rev = args[PVFieldSplitterDnsFqdn::SUBD1_REV].toBool();
	bool subd2_rev = args[PVFieldSplitterDnsFqdn::SUBD2_REV].toBool();
	bool subd3_rev = args[PVFieldSplitterDnsFqdn::SUBD3_REV].toBool();

	set_child_count(tld1 + tld2 + tld3 + subd1 + subd2 + subd3);
	Q_EMIT nchilds_changed_Signal();

	_param_widget = new QWidget();

	auto* layout = new QGridLayout(_param_widget);

	layout->setSizeConstraint(QLayout::SetFixedSize);

	_param_widget->setLayout(layout);
	_param_widget->setObjectName("splitter");

	_split_cb[0] = new QCheckBox("get TLD1");
	_split_cb[0]->setToolTip("get \"com\" in \"www.en.example.com\"");
	_split_cb[0]->setChecked(tld1);

	_split_cb[1] = new QCheckBox("get TLD2");
	_split_cb[1]->setToolTip("get \"example.com\" in \"www.en.example.com\"");
	_split_cb[1]->setChecked(tld2);

	_split_cb[2] = new QCheckBox("get TLD3");
	_split_cb[2]->setToolTip("get \"en.example.com\" in \"www.en.example.com\"");
	_split_cb[2]->setChecked(tld3);

	_split_cb[3] = new QCheckBox("get SUBD1");
	_split_cb[3]->setToolTip("get \"www.en.example\" in \"www.en.example.com\"");
	_split_cb[3]->setChecked(subd1);

	_split_cb[4] = new QCheckBox("get SUBD2");
	_split_cb[4]->setToolTip("get \"www.en\" in \"www.en.example.com\"");
	_split_cb[4]->setChecked(subd2);

	_split_cb[5] = new QCheckBox("get SUBD3");
	_split_cb[5]->setToolTip("get \"www\" in \"www.en.example.com\"");
	_split_cb[5]->setChecked(subd3);

	for (int i = 0; i < 6; ++i) {
		QWidget* w = _split_cb[i];
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
		QWidget* w = _rev_cb[i];
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

	size_t n = 0;

	for (int i = 0; i < 6; ++i) {
		bool s = _split_cb[i]->isChecked();
		if (s) {
			++n;
		}
		if (i >= 3) {
			// need to enable/disable subdomains rev cb
			_rev_cb[i - 3]->setEnabled(s);
		}
	}

	update_args(args);

	get_filter()->set_args(args);

	set_child_count(n);

	Q_EMIT nchilds_changed_Signal();
	Q_EMIT args_changed_Signal();
}

/******************************************************************************
 * PVFilter::PVFieldSplitterDnsFqdnParamWidget::rev_cb_changed
 *****************************************************************************/

void PVFilter::PVFieldSplitterDnsFqdnParamWidget::rev_cb_changed(int)
{
	PVCore::PVArgumentList args = get_filter()->get_args();

	update_args(args);

	get_filter()->set_args(args);

	Q_EMIT args_changed_Signal();
}

/******************************************************************************
 * PVFilter::PVFieldSplitterDnsFqdnParamWidget::update_child_count
 *****************************************************************************/

void PVFilter::PVFieldSplitterDnsFqdnParamWidget::update_args(PVCore::PVArgumentList& args)
{
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
