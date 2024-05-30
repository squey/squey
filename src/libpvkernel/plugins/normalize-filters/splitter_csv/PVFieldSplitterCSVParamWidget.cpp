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

#include "PVFieldSplitterCSVParamWidget.h"
#include "PVFieldSplitterCSV.h"
#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/filter/PVElementFilterByFields.h>

#include <QLabel>
#include <QVBoxLayout>
#include <QSpinBox>

#include <QSpacerItem>
#include <QPushButton>

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterCSVParamWidget::PVFieldSplitterCSVParamWidget
 *
 *****************************************************************************/
PVFilter::PVFieldSplitterCSVParamWidget::PVFieldSplitterCSVParamWidget()
    : PVFieldsSplitterParamWidget(PVFilter::PVFieldsSplitter_p(new PVFieldSplitterCSV()))
{
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterCSVParamWidget::get_param_widget
 *
 *****************************************************************************/
QWidget* PVFilter::PVFieldSplitterCSVParamWidget::get_param_widget()
{
	PVLOG_DEBUG("PVFilter::PVFieldSplitterCSVParamWidget::get_param_widget()     start\n");

	// get args
	PVCore::PVArgumentList l = get_filter()->get_args();

	/*
	 * creating the widget param
	 */
	param_widget = new QWidget();

	// init layout
	auto* layout = new QVBoxLayout(param_widget);
	auto* gridLayout = new QGridLayout();
	param_widget->setLayout(layout);
	param_widget->setObjectName("splitter");

	// title
	auto* label = new QLabel(tr("CSV"), nullptr);
	label->setAlignment(Qt::AlignHCenter);
	layout->addWidget(label);

	// field separator
	auto* separator_label = new QLabel(tr("Field separator:"));
	gridLayout->addWidget(separator_label, 0, 0, Qt::AlignLeft);

	separator_text = new PVWidgets::QKeySequenceWidget();
	separator_text->setClearButtonShow(PVWidgets::QKeySequenceWidget::NoShow);
	separator_text->setKeySequence(QKeySequence((int)l["sep"].toChar().toLatin1()));
	separator_text->setMaxNumKey(1);
	gridLayout->addWidget(separator_text, 0, 1);

	// Quote text
	auto* quote_label = new QLabel("Quote character:");
	gridLayout->addWidget(quote_label, 1, 0, Qt::AlignLeft);

	quote_text = new PVWidgets::QKeySequenceWidget();
	quote_text->setClearButtonShow(PVWidgets::QKeySequenceWidget::NoShow);
	quote_text->setKeySequence(QKeySequence((int)l["quote"].toChar().toLatin1()));
	quote_text->setMaxNumKey(1);
	gridLayout->addWidget(quote_text, 1, 1);

	// field number of col
	auto* col_label = new QLabel(tr("Number of columns:"));
	gridLayout->addWidget(col_label, 2, 0, Qt::AlignLeft);

	// Set default value for number of csv field.
	_child_number_edit = new QSpinBox();
	_child_number_edit->setMaximum(10000);
	_child_number_edit->setValue(get_child_count());
	gridLayout->addWidget(_child_number_edit, 2, 1);
	connect(_child_number_edit, SIGNAL(valueChanged(int)), this, SLOT(updateNChilds()));

	layout->addLayout(gridLayout);
	_recommands_label = new QLabel();
	layout->addWidget(_recommands_label);

	layout->addSpacerItem(new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Expanding));

	// connect(separator_text, SIGNAL(textChanged(const QString &)), this,
	// SLOT(updateSeparator(const QString &)));
	connect(separator_text, SIGNAL(keySequenceChanged(QKeySequence)), this,
	        SLOT(updateSeparator(QKeySequence)));
	connect(quote_text, SIGNAL(keySequenceChanged(QKeySequence)), this,
	        SLOT(updateQuote(QKeySequence)));

	PVLOG_DEBUG("PVFilter::PVFieldSplitterCSVParamWidget::get_param_widget()     end\n");

	return param_widget;
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterCSVParamWidget::get_action_menu
 *
 *****************************************************************************/
QAction* PVFilter::PVFieldSplitterCSVParamWidget::get_action_menu(QWidget* parent)
{
	return new QAction(QString("add CSV Splitter"), parent);
}

void PVFilter::PVFieldSplitterCSVParamWidget::updateSeparator(QKeySequence key)
{
	PVCore::PVArgumentList args = get_filter()->get_args();
	args["sep"] = QChar::fromLatin1(PVWidgets::QKeySequenceWidget::get_ascii_from_sequence(key));
	this->get_filter()->set_args(args);

	Q_EMIT args_changed_Signal();
}

void PVFilter::PVFieldSplitterCSVParamWidget::updateQuote(QKeySequence key)
{
	PVCore::PVArgumentList args = get_filter()->get_args();
	args["quote"] = QChar::fromLatin1(PVWidgets::QKeySequenceWidget::get_ascii_from_sequence(key));
	this->get_filter()->set_args(args);

	Q_EMIT args_changed_Signal();
}

void PVFilter::PVFieldSplitterCSVParamWidget::updateNChilds()
{
	set_child_count(_child_number_edit->value());
	Q_EMIT nchilds_changed_Signal();
}
