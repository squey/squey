/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

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

	QVBoxLayout* layout = new QVBoxLayout(_param_widget);

	_param_widget->setLayout(layout);

	// Label
	QLabel* label = new QLabel(
	    "Identify structural patterns by removing all the alphanumeric characters of a field.\n\n"
	    "The only remaining characters are the following :\n"
	    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~");
	layout->addWidget(label);

	return _param_widget;
}
