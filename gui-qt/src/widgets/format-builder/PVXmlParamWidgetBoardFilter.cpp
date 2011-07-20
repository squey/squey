///! \file PVXmlParamWidgetBoardFilter.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011


#include <PVXmlParamWidgetBoardFilter.h>

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardFilter::PVXmlParamWidgetBoardFilter
 *
 *****************************************************************************/
PVInspector::PVXmlParamWidgetBoardFilter::PVXmlParamWidgetBoardFilter(PVCore::PVXmlTreeNodeDom *pNode) : QWidget() {
    setObjectName("PVXmlParamWidgetBoardFilter");
    node = pNode;
    allocBoardFields();
    draw();
    initValue();
    initConnexion();
    validWidget->setRegEx(exp->text());
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardFilter::~PVXmlParamWidgetBoardFilter
 *
 *****************************************************************************/
PVInspector::PVXmlParamWidgetBoardFilter::~PVXmlParamWidgetBoardFilter() {
    disableConnexion();
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardFilter::allocBoardFields
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardFilter::allocBoardFields() {
  //allocate each field and init them with the saved value
    name = new PVXmlParamWidgetEditorBox(QString("name"), new QVariant(node->getAttribute("name")));
    exp = new PVXmlParamWidgetEditorBox(QString("expression"), new QVariant(node->getDom().attribute("expression",".*")));
    validWidget = new PVXmlParamTextEdit(QString("validator"), QVariant(node->getAttribute("validator")));
    typeOfFilter = new PVXmlParamComboBox("type-of-filter");
    typeOfFilter->addItem("include");
    typeOfFilter->addItem("exclude");
    typeOfFilter->select(node->getAttribute("type-of-filter"));
    buttonNext = new QPushButton("Next");
    buttonNext->setShortcut(QKeySequence(Qt::Key_Return));
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardFilter::disableConnexion
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardFilter::disableConnexion() {
    disconnect(name, SIGNAL(textChanged(const QString&)), this, SLOT(slotSetValues()));
    disconnect(exp, SIGNAL(textChanged(const QString&)), validWidget, SLOT(setRegEx(const QString &)));
    disconnect(exp, SIGNAL(textChanged(const QString&)), this, SLOT(slotSetValues()));
    disconnect(validWidget, SIGNAL(textChanged()), this, SLOT(slotSetValues()));
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardFilter::disAllocBoardFields
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardFilter::disAllocBoardFields() {

}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardFilter::draw
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardFilter::draw() {
    QVBoxLayout *qv=new QVBoxLayout();
    
    qv->addWidget(new QLabel("Filter name"));
    qv->addWidget(name);
    qv->addWidget(typeOfFilter);
    qv->addWidget(new QLabel("Expression"));
    qv->addWidget(exp);
    qv->addWidget(new QLabel("Expression validator"));
    qv->addWidget(validWidget);
    qv->addWidget(buttonNext);
    
    setLayout(qv);
    
    //name->textCursor().movePosition(QTextCursor::Start);
}


/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardFilter::getWidgetToFocus
 *
 *****************************************************************************/
QWidget *PVInspector::PVXmlParamWidgetBoardFilter::getWidgetToFocus(){
  return (QWidget *)name;
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardFilter::initConnexion
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardFilter::initConnexion() {
    connect(name, SIGNAL(textChanged(const QString&)), this, SLOT(slotSetValues()));
    connect(name, SIGNAL(textChanged(const QString&)), this, SLOT(slotVerifRegExpInName()));
    connect(exp, SIGNAL(textChanged(const QString&)), validWidget, SLOT(setRegEx(const QString &)));
    connect(exp, SIGNAL(textChanged(const QString&)), this, SLOT(slotSetValues()));
    connect(validWidget, SIGNAL(textChanged()), this, SLOT(slotSetValues()));
    connect(typeOfFilter, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(slotSetValues()));
    connect(buttonNext, SIGNAL(clicked()), this, SLOT(slotEmitNext()));
}


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardFilter::initValue
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardFilter::initValue() {
    
}


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardFilter::slotEmitNext
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardFilter::slotEmitNext(){
  emit signalEmitNext();
}


/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardFilter::slotSetValues
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardFilter::slotSetValues(){//called when we modifi something.
    node->setAttribute(QString("name"),name->text());
    node->setAttribute(QString("expression"),exp->text());
    node->setAttribute(QString("validator"),validWidget->getVal().toString());
    node->setAttribute(QString("type-of-filter"),typeOfFilter->val().toString());

    emit signalRefreshView();
}


/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardFilter::slotVerifRegExpInName
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardFilter::slotVerifRegExpInName(){
  //char we want to detecte in the name  
  QRegExp reg(".*(\\*|\\[|\\{|\\]|\\}).*");
    if (reg.exactMatch(name->text())) {
      //create and open the confirm box
        QDialog confirm(this);
        QVBoxLayout vb;
        confirm.setLayout(&vb);
        vb.addWidget(new QLabel("you are maybe writting a regexp in the name field. Do you want to write it in the expression field ?"));
        QHBoxLayout bas;
        vb.addLayout(&bas);
        QPushButton no("No");
        bas.addWidget(&no);
        QPushButton yes("Yes");
        bas.addWidget(&yes);

        //connect the response button
        connect(&no, SIGNAL(clicked()), &confirm, SLOT(reject()));
        connect(&yes, SIGNAL(clicked()), &confirm, SLOT(accept()));

        //if confirmed then apply
        if (confirm.exec()) {
            exp->setText(name->text());
            name->setText("");
        }
    }
}


