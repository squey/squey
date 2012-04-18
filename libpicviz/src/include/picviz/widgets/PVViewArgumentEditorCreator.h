//! \file PVArgumentListWidget.h
//! $Id: PVArgumentEditorCreator.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVARGUMENTEDITORCREATOR_H
#define PVARGUMENTEDITORCREATOR_H

#include <QByteArray>
#include <QItemEditorCreatorBase>
#include <QWidget>

#include <pvkernel/core/general.h>
#include <picviz/PVView.h>

namespace PVWidgets {

// Inspired by QStandardItemEditorCreator
// Reuse the Q_PROPERTY macros
template <class T>
class PVViewArgumentEditorCreator: public QItemEditorCreatorBase
{
public:
    inline PVViewArgumentEditorCreator(Picviz::PVView const& view)
        : propertyName(T::staticMetaObject.userProperty().name()), _view(view)
    {}
    inline QWidget *createWidget(QWidget *parent) const { return new T(_view, parent); }
    inline QByteArray valuePropertyName() const { return propertyName; }

private:
    QByteArray propertyName;
	Picviz::PVView const& _view;
};

}

#endif	/* PVARGUMENTEDITORCREATOR_H */
