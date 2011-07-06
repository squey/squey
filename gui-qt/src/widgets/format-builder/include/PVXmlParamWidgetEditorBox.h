//! \file PVXmlParamWidgetEditorBox.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVXMLPARAMWIDGETEDITORBOX_H
#define	PVXMLPARAMWIDGETEDITORBOX_H
#include <QLineEdit>
#include <QVariant>
#include <QString>
#include <QTextEdit>
#include <QPushButton>
#include <QMessageBox>
#include <iostream>


namespace PVInspector{
class PVXmlParamWidgetEditorBox:public QLineEdit {
public:
    PVXmlParamWidgetEditorBox();
    PVXmlParamWidgetEditorBox(QString name,QVariant *var);
    virtual ~PVXmlParamWidgetEditorBox();
    QVariant val();
    void setVal(const QVariant &val);
private:
        QVariant *variable;

};
}
#endif	/* PVXMLPARAMWIDGETEDITORBOX_H */

