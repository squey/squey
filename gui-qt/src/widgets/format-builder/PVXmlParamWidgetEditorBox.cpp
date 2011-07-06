///! \file PVXmlParamWidgetEditorBox.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011
#include <PVXmlParamWidgetEditorBox.h>



/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetEditorBox::PVXmlParamWidgetEditorBox
 *
 *****************************************************************************/
PVInspector::PVXmlParamWidgetEditorBox::PVXmlParamWidgetEditorBox(QString pName,QVariant *var):QLineEdit() {
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


