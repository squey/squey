/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVFieldSplitterMacAddressParamWidget.h"
#include "PVFieldSplitterMacAddress.h"

#include <QAction>

/******************************************************************************
 * PVFilter::PVFieldSplitterMacAddressParamWidget::PVFieldSplitterCSVParamWidget
 *****************************************************************************/

PVFilter::PVFieldSplitterMacAddressParamWidget::PVFieldSplitterMacAddressParamWidget()
    : PVFieldsSplitterParamWidget(PVFilter::PVFieldsSplitter_p(new PVFieldSplitterMacAddress()))
{
}

/******************************************************************************
 * PVFilter::PVFieldSplitterMacAddressParamWidget::get_action_menu
 *****************************************************************************/

QAction* PVFilter::PVFieldSplitterMacAddressParamWidget::get_action_menu(QWidget* parent)
{
	return new QAction(QString("add Mac Address Splitter"), parent);
}

/******************************************************************************
 * PVFilter::PVFieldSplitterMacAddressParamWidget::get_param_widget
 *****************************************************************************/

QWidget* PVFilter::PVFieldSplitterMacAddressParamWidget::get_param_widget()
{
	set_child_count(2);
	emit nchilds_changed_Signal();

	return new QWidget();
}
