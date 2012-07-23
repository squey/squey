/**
 * \file PVXmlParamList.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVXMLPARAMLIST_H
#define	PVXMLPARAMLIST_H

#include <QListWidget>
#include <QString>
#include <QStringList>

namespace PVInspector {

class PVXmlParamList: public QListWidget {
    Q_OBJECT
public:
    PVXmlParamList(QString const& name);
public:
	void setItems(QStringList const& l);
    QStringList selectedList();
    void select(QStringList const& l);
	QString const& name() const { return _name; }
protected:
	QString _name;
};

}

#endif
