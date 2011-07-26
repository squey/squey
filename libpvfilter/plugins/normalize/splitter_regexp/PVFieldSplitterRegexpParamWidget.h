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
#include <QTextEdit>
#include <QTableWidget>
#include <QPushButton>
#include <QStringList>
#include <QRegExp>

namespace PVFilter {


class PVFieldSplitterRegexpParamWidget: public PVFieldsSplitterParamWidget {
    Q_OBJECT;
private:
    QAction* action_menu;
    QWidget* param_widget;
    int id;
    
    //widget showed
    QLineEdit *expression_lineEdit;
    QLabel *child_count_text;
    QTextEdit *validator_textEdit;
    QTableWidget *table_validator_TableWidget;
    QPushButton *btn_apply;
    
    bool expressionChanged;
    
    void initWidget();
    
    

public:
    PVFieldSplitterRegexpParamWidget();
    QAction* get_action_menu();
    QWidget* get_param_widget();
    
    void set_id(int id_param) {
        id = id_param;
    }
    
    
    void update_data_display();
    
    
    
    CLASS_REGISTRABLE_NOCOPY(PVFieldSplitterRegexpParamWidget)
public slots:
    void slotUpdateTableValidator();
    void slotExpressionChanged();

signals:
    void data_changed();
    void signalRefreshView();
};

}

#endif
