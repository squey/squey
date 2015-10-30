/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <PVXmlParamWidgetEditorBox.h>

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetEditorBox::PVXmlParamWidgetEditorBox
 *
 *****************************************************************************/
PVInspector::PVXmlParamWidgetEditorBox::PVXmlParamWidgetEditorBox(QString pName,QVariant *var):QLineEdit() {
    setObjectName("PVXmlParamWidgetEditorBox");    
    variable = var;
    setObjectName(pName);
    setText(variable->toString());
    emit textChanged(variable->toString());
}





/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetEditorBox::~PVXmlParamWidgetEditorBox
 *
 *****************************************************************************/
PVInspector::PVXmlParamWidgetEditorBox::~PVXmlParamWidgetEditorBox() {
}




/******************************************************************************
 *
 * QVariant PVInspector::PVXmlParamWidgetEditorBox::val
 *
 *****************************************************************************/
QVariant PVInspector::PVXmlParamWidgetEditorBox::val(){
    return this->displayText();
}




/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetEditorBox::setVal
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetEditorBox::setVal(const QVariant &val){
    variable = new QVariant(val);
    setText(variable->toString());
    emit textChanged(variable->toString());
}


