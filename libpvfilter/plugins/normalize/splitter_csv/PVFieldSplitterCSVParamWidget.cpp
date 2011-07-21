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

#if 0
PVFilter::PVFieldSplitterCSVParamWidget::PVFieldSplitterCSVParamWidget(const PVFieldSplitterCSVParamWidget& /*src*/) :
        QObject(), PVFieldsSplitterParamWidget(PVFilter::PVFieldsSplitter_p(new PVFieldSplitterCSV()))
{
        init();
}
#endif

void PVFilter::PVFieldSplitterCSVParamWidget::init()
{
        PVLOG_DEBUG("init PVFieldSplitterCSVParamWidget\n");
        /*
         * action menu init
         */
        action_menu = new QAction(QString("add CSV Splitter"),NULL);
        
        /*
         * creating the widget param
         */
        param_widget = new QWidget();
        //init layout
        QBoxLayout* layout = new QBoxLayout(param_widget);
        param_widget->setLayout(layout);
        QLabel* label = new QLabel(tr("name"),layout);
        layout->addWidget(label);
}
/******************************************************************************
 *
 * PVFilter::PVFieldSplitterCSVParamWidget::get_param_widget
 *
 *****************************************************************************/
QWidget* PVFilter::PVFieldSplitterCSVParamWidget::get_param_widget()
{
        return param_widget;
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

