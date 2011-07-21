#ifndef PVFIELDSPLITTERCSVPARAMWIDGET_H
#define PVFIELDSPLITTERCSVPARAMWIDGET_H

#include <pvcore/general.h>
#include <pvfilter/PVFieldsFilterParamWidget.h>
#include <boost/shared_ptr.hpp>

#include <QObject>
#include <QAction>

namespace PVFilter {

class PVFieldSplitterCSVParamWidget: public QObject, public PVFieldsSplitterParamWidget {
    Q_OBJECT;
public:
    
    PVFieldSplitterCSVParamWidget();
    PVFieldSplitterCSVParamWidget(const PVFieldSplitterCSVParamWidget& src);

private:
    QAction* action_menu;
    QWidget* param_widget;
    int id;

public:
    QWidget* get_param_widget();
    QAction* get_action_menu();
    QString get_xml_tag() { return QString("splitter"); }

    void set_id(int id_param) {
        id = id_param;
    }
private:
    void init();
    
    CLASS_REGISTRABLE(PVFieldSplitterCSVParamWidget)
    

};

}

#endif
