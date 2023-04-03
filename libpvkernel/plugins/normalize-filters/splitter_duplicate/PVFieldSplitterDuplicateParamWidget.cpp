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

#include "PVFieldSplitterDuplicateParamWidget.h"
#include "PVFieldDuplicate.h"
#include <pvkernel/filter/PVFieldsFilter.h>

#include <QLabel>
#include <QHBoxLayout>
#include <QSpacerItem>

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterDuplicateParamWidget::PVFieldSplitterCSVParamWidget
 *
 *****************************************************************************/
PVFilter::PVFieldSplitterDuplicateParamWidget::PVFieldSplitterDuplicateParamWidget()
    : PVFieldsSplitterParamWidget(PVFilter::PVFieldsSplitter_p(new PVFieldDuplicate()))
{
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterDuplicateParamWidget::get_action_menu
 *
 *****************************************************************************/
QAction* PVFilter::PVFieldSplitterDuplicateParamWidget::get_action_menu(QWidget* parent)
{
	return new QAction(QString("add Duplicate Splitter"), parent);
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterDuplicateParamWidget::get_param_widget
 *
 *****************************************************************************/
QWidget* PVFilter::PVFieldSplitterDuplicateParamWidget::get_param_widget()
{
	PVLOG_DEBUG("PVFilter::PVFieldSplitterDuplicateParamWidget::get_param_widget()     start\n");

	// get args
	PVCore::PVArgumentList l = get_filter()->get_args();

	uint32_t n = l["n"].toUInt();
	updateNChilds(n);

	_param_widget = new QWidget();

	auto* layout = new QHBoxLayout(_param_widget);

	_param_widget->setLayout(layout);
	_param_widget->setObjectName("splitter");

	// title
	auto* duplications_label = new QLabel(tr("Number of duplications:"));
	layout->addWidget(duplications_label);

	// Spin box
	_duplications_spin_box = new QSpinBox();
	_duplications_spin_box->setMinimum(2);
	_duplications_spin_box->setValue(n);
	layout->addWidget(_duplications_spin_box);

	connect(_duplications_spin_box, SIGNAL(valueChanged(int)), this, SLOT(updateNChilds(int)));

	return _param_widget;
}

void PVFilter::PVFieldSplitterDuplicateParamWidget::updateNChilds(int n)
{
	PVCore::PVArgumentList args = get_filter()->get_args();
	args["n"] = n;
	get_filter()->set_args(args);
	set_child_count(n);
	Q_EMIT args_changed_Signal();
	Q_EMIT nchilds_changed_Signal();
}
