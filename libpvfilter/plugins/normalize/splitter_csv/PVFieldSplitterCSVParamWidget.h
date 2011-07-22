#ifndef PVFIELDSPLITTERCSVPARAMWIDGET_H
#define PVFIELDSPLITTERCSVPARAMWIDGET_H

#include <pvcore/general.h>
#include <pvfilter/PVFieldsFilterParamWidget.h>
#include <boost/shared_ptr.hpp>

#include <QBoxLayout>
#include <QLabel>
#include <QObject>
#include <QAction>
#include <QLineEdit>

namespace PVFilter {

class PVFieldSplitterCSVParamWidget: public QObject, public PVFieldsSplitterParamWidget {
    Q_OBJECT;
public:
    
    PVFieldSplitterCSVParamWidget();
    //PVFieldSplitterCSVParamWidget(const PVFieldSplitterCSVParamWidget& src);

private:
    QAction* action_menu;
    QWidget* param_widget;
    int id,child_count;
    QLineEdit *child_number_edit;

public:
    PVCore::PVArgumentList get_default_argument(){
        PVCore::PVArgumentList args;
        args["sep"]=QVariant(";");
        return args;
    }
    QWidget* get_param_widget();
    QAction* get_action_menu();
    QString get_xml_tag() { return QString("splitter"); }

    void set_id(int id_param) {
        id = id_param;
    }
    void set_child_count(int count){
        child_count = count;
    }
    int get_child_new_num() {
        return child_count;
    }
    
    QObject* get_as_qobject(){return this;}
private:
    void init();

    
    CLASS_REGISTRABLE_NOCOPY(PVFieldSplitterCSVParamWidget)
    
public slots:
    void updateSeparator(const QString &sep){
        PVCore::PVArgumentList args;
        args["sep"]=QVariant(sep);
        this->get_filter()->set_args(args);
        
        emit data_changed();
    }
    void updateChildCount(){
        assert(child_number_edit);
        set_child_count(child_number_edit->text().toInt());
        emit data_changed();
    }

    
    signals:
    void data_changed();
    void signalRefreshView();
};

}

#endif
