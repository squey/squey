//! \file PVXmlParamComboBox.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVXMLPARAMCOMBOBOX_H
#define	PVXMLPARAMCOMBOBOX_H
#include <QComboBox>
#include <QString>
#include <QVariant>
namespace PVInspector{
class PVXmlParamComboBox: public QComboBox {
    Q_OBJECT
public:
    PVXmlParamComboBox(QString name);
    virtual ~PVXmlParamComboBox();
    QVariant val();
    void select(QString title);
private:

};
}
#endif	/* PVXMLPARAMCOMBOBOX_H */

