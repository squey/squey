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
        init();
}

PVFilter::PVFieldSplitterCSVParamWidget::PVFieldSplitterCSVParamWidget(const PVFieldSplitterCSVParamWidget& /*src*/) :
        QObject(), PVFieldsSplitterParamWidget(PVFilter::PVFieldsSplitter_p(new PVFieldSplitterCSV()))
{
        init();
}

void PVFilter::PVFieldSplitterCSVParamWidget::init()
{
        PVLOG_DEBUG("init PVFieldSplitterCSVParamWidget\n");
        action_menu = new QAction(QString("add CSV Splitter"),NULL);
}
/******************************************************************************
 *
 * PVFilter::PVFieldSplitterCSVParamWidget::get_param_widget
 *
 *****************************************************************************/
QWidget* PVFilter::PVFieldSplitterCSVParamWidget::get_param_widget()
{
        QWidget* w = new QWidget();
	return w;
}



/******************************************************************************
 *
 * PVFilter::PVFieldSplitterCSVParamWidget::get_action_menu
 *
 *****************************************************************************/
QAction* PVFilter::PVFieldSplitterCSVParamWidget::get_action_menu(){
        assert(action_menu);
        return action_menu;
}

