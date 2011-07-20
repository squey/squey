#include "PVFieldSplitterCSVParamWidget.h"
#include "PVFieldSplitterCSV.h"
#include <pvfilter/PVFieldsFilter.h>




/******************************************************************************
 *
 * PVFilter::PVFieldSplitterCSVParamWidget::PVFieldSplitterCSVParamWidget
 *
 *****************************************************************************/
PVFilter::PVFieldSplitterCSVParamWidget::PVFieldSplitterCSVParamWidget() :
	PVFieldsSplitterParamWidget(PVFilter::PVFieldsSplitter_p(new PVFieldSplitterCSV()))
{
        action_menu = new QAction(QString("add CSV Splitter"),NULL);
}


/******************************************************************************
 *
 * PVFilter::PVFieldSplitterCSVParamWidget::get_param_widget
 *
 *****************************************************************************/
QWidget* PVFilter::PVFieldSplitterCSVParamWidget::get_param_widget()
{
	return NULL;
}



/******************************************************************************
 *
 * PVFilter::PVFieldSplitterCSVParamWidget::get_action_menu
 *
 *****************************************************************************/
QAction* PVFilter::PVFieldSplitterCSVParamWidget::get_action_menu(){
        return action_menu;
}
