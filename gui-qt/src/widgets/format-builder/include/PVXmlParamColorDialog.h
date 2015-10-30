/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

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

