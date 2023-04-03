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

#include "PVFieldConverterStructParamWidget.h"
#include "PVFieldConverterStruct.h"

#include <QAction>
#include <QVBoxLayout>
#include <QLabel>

/******************************************************************************
 *
 * PVFilter::PVFieldConverterStructParamWidget::PVFieldConverterStructParamWidget
 *
 *****************************************************************************/
PVFilter::PVFieldConverterStructParamWidget::PVFieldConverterStructParamWidget()
    : PVFieldsConverterParamWidget(PVFilter::PVFieldsConverter_p(new PVFieldConverterStruct()))
{
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterStructParamWidget::get_action_menu
 *
 *****************************************************************************/
QAction* PVFilter::PVFieldConverterStructParamWidget::get_action_menu(QWidget* parent)
{
	return new QAction(QString("add Struct Converter"), parent);
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterStructParamWidget::get_param_widget
 *
 *****************************************************************************/
QWidget* PVFilter::PVFieldConverterStructParamWidget::get_param_widget()
{
	PVLOG_DEBUG("PVFilter::PVFieldConverterStructParamWidget::get_param_widget()     start\n");

	PVCore::PVArgumentList args = get_filter()->get_args();

	_param_widget = new QWidget;

	auto* layout = new QVBoxLayout(_param_widget);

	_param_widget->setLayout(layout);

	// Label
	auto* label = new QLabel(
	    "Identify structural patterns by removing all the alphanumeric characters of a field.\n\n"
	    "The only remaining characters are the following :\n"
	    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~");
	layout->addWidget(label);

	return _param_widget;
}
