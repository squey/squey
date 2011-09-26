///! \file PVXmlParamTextEdit.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#include <PVXmlParamTextEdit.h>
#include <PVXmlRegValidatorHighLight.h>
#include <PVXmlTimeValidatorHighLight.h>

#include <QSizePolicy>


/******************************************************************************
 *
 * PVInspector::PVXmlParamTextEdit::PVXmlParamTextEdit
 *
 *****************************************************************************/
PVInspector::PVXmlParamTextEdit::PVXmlParamTextEdit(QString pName,QVariant var):
	QTextEdit()
{
    setObjectName(pName);
    variable = var.toString();
    setText(variable);
    
    typeOfTextEdit = text;
    editing = true;

	QSizePolicy sp(QSizePolicy::Expanding, QSizePolicy::Maximum);
	sp.setHeightForWidth(sizePolicy().hasHeightForWidth());
	setSizePolicy(sp);
	setMaximumHeight(70);

    connect(this,SIGNAL(textChanged()),this,SLOT(slotHighLight()));
}



/******************************************************************************
 *
 * PVInspector::PVXmlParamTextEdit::~PVXmlParamTextEdit
 *
 *****************************************************************************/
PVInspector::PVXmlParamTextEdit::~PVXmlParamTextEdit() {
    
    highlight->deleteLater();
}




/******************************************************************************
 *
 * PVInspector::PVXmlParamTextEdit::setRegEx
 *
 *****************************************************************************/
void PVInspector::PVXmlParamTextEdit::setRegEx(const QString &regStr){
    if(editing){
        highlight =new PVXmlRegValidatorHighLight((PVXmlParamTextEdit*)this);//we are in regexp validator case
	editing=false;
    }
    typeOfTextEdit = regexpValid;//define type as a regexp validator
    highlight->setRegExp(regStr);
    highlight->rehighlight();
}
    



/******************************************************************************
 *
 * PVInspector::PVXmlParamTextEdit::slotHighLight
 *
 *****************************************************************************/
void PVInspector::PVXmlParamTextEdit::slotHighLight(){
    if(typeOfTextEdit == dateValid){
     //qDebug()<<"PVXmlParamTextEdit::slotHighLight for axis";
      //timeValid->rehighlight();
    }
}




/******************************************************************************
 *
 * QVariant PVInspector::PVXmlParamTextEdit::getVal
 *
 *****************************************************************************/
QVariant PVInspector::PVXmlParamTextEdit::getVal()const{
    //variable=QString(toPlainText());
    return QVariant(toPlainText());
}




/******************************************************************************
 *
 * PVInspector::PVXmlParamTextEdit::setVal
 *
 *****************************************************************************/
void  PVInspector::PVXmlParamTextEdit::setVal(const QString &val){
    variable=val;
    setText(variable);
}



/******************************************************************************
 *
 * PVInspector::PVXmlParamTextEdit::validDateFormat
 *
 *****************************************************************************/
void PVInspector::PVXmlParamTextEdit::validDateFormat(const QStringList & pFormat){
  if(editing){
    timeValid = new PVXmlTimeValidatorHighLight(this,format);//we are in time validator case
    editing=false;
  }
  typeOfTextEdit = dateValid;//define type as a date validator.
  format = pFormat;
  timeValid->setDateFormat(format);//set the format to validate.
  timeValid->rehighlight();//update color for validated line
}
