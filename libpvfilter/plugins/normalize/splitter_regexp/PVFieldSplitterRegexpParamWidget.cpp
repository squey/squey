#include "PVFieldSplitterRegexpParamWidget.h"
#include "PVFieldSplitterRegexp.h"
#include <pvfilter/PVFieldsFilter.h>


#include <QLabel>
#include <QVBoxLayout>

#include <QSpacerItem>
#include <QPushButton>

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterRegexpParamWidget::PVFieldSplitterRegexpParamWidget
 *
 *****************************************************************************/
PVFilter::PVFieldSplitterRegexpParamWidget::PVFieldSplitterRegexpParamWidget() :
	PVFieldsSplitterParamWidget(PVFilter::PVFieldsSplitter_p(new PVFieldSplitterRegexp()))
{
    PVLOG_DEBUG("constructor PVFieldSplitterRegexpParamWidget\n");
    action_menu = new QAction(QString("add RegExp Splitter"),NULL);
 
    expressionChanged = false;
}


/******************************************************************************
 *
 * PVFilter::PVFieldSplitterRegexpParamWidget::initWidget
 *
 *****************************************************************************/
void PVFilter::PVFieldSplitterRegexpParamWidget::initWidget(){
    PVCore::PVArgumentList l =  get_filter()->get_args();
    expression_lineEdit = new QLineEdit(l["regexp"].toString());
    child_count_text = new QLabel("child count");
    validator_textEdit = new QTextEdit(get_data().join("\n"));
    table_validator_TableWidget = new QTableWidget();
    btn_apply = new QPushButton(tr("Apply"));
}


/******************************************************************************
 *
 * PVFilter::PVFieldSplitterRegexpParamWidget::get_action_menu
 *
 *****************************************************************************/
QAction* PVFilter::PVFieldSplitterRegexpParamWidget::get_action_menu() {
    PVLOG_DEBUG("get action PVFieldSplitterRegexpParamWidget\n");
    assert(action_menu);
    return action_menu;
    expressionChanged = false;
}


/******************************************************************************
 *
 * PVFilter::PVFieldSplitterRegexpParamWidget::get_param_widget
 *
 *****************************************************************************/
QWidget* PVFilter::PVFieldSplitterRegexpParamWidget::get_param_widget()
{
    PVLOG_DEBUG("get_param_widget PVFieldSplitterRegexpParamWidget\n");
    initWidget();
    
    param_widget = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(param_widget);
    param_widget->setLayout(layout);
    param_widget->setObjectName("splitter");
    
    //expression
    layout->addWidget(new QLabel("Expression"));
    layout->addWidget(expression_lineEdit);
    
    //child count
    layout->addWidget(child_count_text);
    
    //validator zone
    layout->addWidget(validator_textEdit);
    layout->addWidget(table_validator_TableWidget);
    layout->addWidget(btn_apply);
    
    
    connect(expression_lineEdit,SIGNAL(textChanged(QString)),this,SLOT(slotExpressionChanged()));
    connect(validator_textEdit,SIGNAL(textChanged()),this,SLOT(slotUpdateTableValidator()));

    update_data_display();
    slotUpdateTableValidator();
    return param_widget;
}
/******************************************************************************
 *
 * PVFilter::PVFieldSplitterRegexpParamWidget::slotExpressionChanged
 *
 *****************************************************************************/
void PVFilter::PVFieldSplitterRegexpParamWidget::slotExpressionChanged(){
    PVLOG_DEBUG("slotExpressionChanged() : PVFieldSplitterRegexpParamWidget: %x\n", this);
    expressionChanged = true;
    //child count
    QRegExp reg = QRegExp(expression_lineEdit->text());
    PVLOG_DEBUG("set_child_count(reg.captureCount()); %d\n",reg.captureCount());
    set_child_count(reg.captureCount());
    
    PVCore::PVArgumentList l;
    l["regexp"] = PVCore::PVArgument(expression_lineEdit->text());
    get_filter()->set_args(l);
    
    emit args_changed_Signal();
    emit nchilds_changed_Signal();
    
    slotUpdateTableValidator();
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterRegexpParamWidget::slotUpdateTableValidator
 *
 *****************************************************************************/
void PVFilter::PVFieldSplitterRegexpParamWidget::slotUpdateTableValidator(){
    PVLOG_DEBUG("slotUpdateTableValidator() : PVFieldSplitterRegexpParamWidget\n");
    QRegExp reg = QRegExp(expression_lineEdit->text());
    //update the number of column
    reg.indexIn(validator_textEdit->toPlainText(), 0);
    table_validator_TableWidget->setColumnCount(reg.captureCount());
    


    //feed each line with the matching in text zone.
    QStringList myText = validator_textEdit->toPlainText().split("\n");
    PVLOG_DEBUG("child count :\n%d\n",myText.count());
    table_validator_TableWidget->setRowCount(myText.count());
    //updateHeaderTable();//can't access to the node ...
    for (int line = 0; line < myText.count(); line++) {//for each line...
        QString myLine = myText.at(line);
        //if (reg.indexIn(myLine, 0)) {
            for (int cap = 0; cap < reg.captureCount(); cap++) {//for each column (regexp selection)...
                reg.indexIn(myLine, 0);
                table_validator_TableWidget->setItem(line, cap, new QTableWidgetItem(reg.cap(cap + 1)));
                int width = 12 + (8 * reg.cap(cap + 1).length());
                if (width > table_validator_TableWidget->columnWidth(cap)) {
                    table_validator_TableWidget->setColumnWidth(cap, width); //update the size
                }
            }
        //}
    }
    table_validator_TableWidget->setContentsMargins(3, 0, 3, 0);
}





/******************************************************************************
 *
 * PVFilter::PVFieldSplitterRegexpParamWidget::update_data_display
 *
 *****************************************************************************/
void PVFilter::PVFieldSplitterRegexpParamWidget::update_data_display(){
    PVLOG_DEBUG("update_data_display() : PVFieldSplitterRegexpParamWidget\n");
    validator_textEdit->setText(get_data().join("\n"));
}