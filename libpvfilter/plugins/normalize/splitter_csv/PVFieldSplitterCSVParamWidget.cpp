#include "PVFieldSplitterCSVParamWidget.h"
#include "PVFieldSplitterCSV.h"
#include <pvfilter/PVFieldsFilter.h>


#include <QLabel>
#include <QVBoxLayout>

#include <QSpacerItem>
#include <QPushButton>

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


/******************************************************************************
 *
 * PVFilter::PVFieldSplitterCSVParamWidget::init
 *
 *****************************************************************************/
void PVFilter::PVFieldSplitterCSVParamWidget::init()
{
        PVLOG_DEBUG("init PVFieldSplitterCSVParamWidget\n");
        
        action_menu = new QAction(QString("add CSV Splitter"),NULL);
        
        child_count = 0;
        
}
/******************************************************************************
 *
 * PVFilter::PVFieldSplitterCSVParamWidget::get_param_widget
 *
 *****************************************************************************/
QWidget* PVFilter::PVFieldSplitterCSVParamWidget::get_param_widget()
{
        PVLOG_DEBUG("PVFilter::PVFieldSplitterCSVParamWidget::get_param_widget()     start\n");
        
        //get args
        PVCore::PVArgumentList l =  get_filter()->get_args();
        
        
        /*
         * creating the widget param
         */
        param_widget = new QWidget();
        //init layout
        QVBoxLayout* layout = new QVBoxLayout(param_widget);
        param_widget->setLayout(layout);
        param_widget->setObjectName("splitter");
        //title
        QLabel* label = new QLabel(tr("CSV"),NULL);
        label->setAlignment(Qt::AlignHCenter);
        layout->addWidget(label);
        //field separator
        QLabel* separator_label = new QLabel(tr("separator"),NULL);
        separator_label->setAlignment(Qt::AlignLeft);
        layout->addWidget(separator_label);
        QLineEdit* separator_text = new QLineEdit(l["sep"].toString());
        separator_text->setAlignment(Qt::AlignHCenter);
        separator_text->setMaxLength(1);
        layout->addWidget(separator_text);
        //field number of col
        QLabel* col_label = new QLabel(tr("column number"),NULL);
        col_label->setAlignment(Qt::AlignLeft);
        layout->addWidget(col_label);
        PVLOG_DEBUG("there is %d child(ren)\n",child_count);
        child_number_edit = new QLineEdit(QString("%1").arg(child_count));
        child_number_edit->setAlignment(Qt::AlignHCenter);
        layout->addWidget(child_number_edit);
        //apply buttom
        QPushButton *btn = new QPushButton(tr("Apply"));
        btn->setShortcut(Qt::Key_Return);
        layout->addWidget(btn);
        
        layout->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Expanding, QSizePolicy::Expanding));
        
        
        connect(separator_text,SIGNAL(textChanged(const QString &)),this,SLOT(updateSeparator(const QString &)));
        connect(btn,SIGNAL(clicked()),this,SLOT(updateChildCount()));

        
        PVLOG_DEBUG("PVFilter::PVFieldSplitterCSVParamWidget::get_param_widget()     end\n");
        
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

