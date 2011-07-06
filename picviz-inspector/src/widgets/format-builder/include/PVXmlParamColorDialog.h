//! \file PVXmlParamColorDialog.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVXMLPARAMCOLORDIALOG_H
#define	PVXMLPARAMCOLORDIALOG_H

#include <QPushButton>
#include <QColorDialog>
#include <QString>
#include <QObject>
#include <QDebug>
#include <QColor>
namespace PVInspector{
class PVXmlParamColorDialog : public QPushButton {
    Q_OBJECT
public:
    PVXmlParamColorDialog(QString name, QString color, QWidget* parent=0);
    virtual ~PVXmlParamColorDialog();
    void setColor(QString);
    QString getColor();
private:
    QString color;
    QWidget *parent;

public slots:
    void chooseColor();
    signals:
    void changed();
    
};
}
#endif	/* PVXMLPARAMCOLORDIALOG_H */

