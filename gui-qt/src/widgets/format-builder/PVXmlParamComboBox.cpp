///! \file PVXmlParamComboBox.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#include <PVXmlParamComboBox.h>



/******************************************************************************
 *
 * PVInspector::PVXmlParamComboBox::PVXmlParamComboBox
 *
 *****************************************************************************/
PVInspector::PVXmlParamComboBox::PVXmlParamComboBox(QString name):QComboBox() {
    setObjectName(name);
}





/******************************************************************************
 *
 * PVInspector::PVXmlParamComboBox::~PVXmlParamComboBox
 *
 *****************************************************************************/
PVInspector::PVXmlParamComboBox::~PVXmlParamComboBox() {
}



/******************************************************************************
 *
 * QVariant PVInspector::PVXmlParamComboBox::val
 *
 *****************************************************************************/
QVariant PVInspector::PVXmlParamComboBox::val(){
    //return the current selected item title.
    return this->currentText();
}



/******************************************************************************
 *
 * void PVInspector::PVXmlParamComboBox::select
 *
 *****************************************************************************/
void PVInspector::PVXmlParamComboBox::select(QString title){
    for(int i=0;i<count();i++){//for each item...
        if(itemText(i)==title){//...if the title match...
	  this->setCurrentIndex(i);//...select it.
	}
    }
}


