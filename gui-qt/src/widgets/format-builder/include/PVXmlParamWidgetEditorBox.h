/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

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

