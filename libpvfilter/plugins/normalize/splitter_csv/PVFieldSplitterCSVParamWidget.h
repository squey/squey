#ifndef PVFIELDSPLITTERCSVPARAMWIDGET_H
#define PVFIELDSPLITTERCSVPARAMWIDGET_H

#include <pvcore/general.h>
#include <pvfilter/PVFieldsFilterParamWidget.h>
#include <boost/shared_ptr.hpp>

#include <QAction>

namespace PVFilter {

class PVFieldSplitterCSVParamWidget: public PVFieldsSplitterParamWidget
{
public:
	PVFieldSplitterCSVParamWidget();
    
private:
    QAction* action_menu;
    QWidget* param_widget;
    
public:
	QWidget* get_param_widget();
    QAction* get_action_menu();


	CLASS_REGISTRABLE(PVFieldSplitterCSVParamWidget)
            

};

}

#endif
