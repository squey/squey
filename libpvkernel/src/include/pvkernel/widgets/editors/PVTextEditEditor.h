/**
 * \file PVTextEditEditor.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PVTEXTEDITEDITOR_H
#define PVCORE_PVTEXTEDITEDITOR_H

#include <QWidget>
#include <QTextEdit>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVTextEditType.h>

namespace PVWidgets {

/**
 * \class PVTextEditEditor
 */
class PVTextEditEditor: public QWidget
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVTextEditType _text READ get_text WRITE set_text USER true)

public:
	PVTextEditEditor(QWidget *parent = 0);

public:
	PVCore::PVTextEditType get_text() const;
	void set_text(PVCore::PVTextEditType const& text);

/* protected: */
/* 	virtual bool eventFilter(QObject *object, QEvent *event); */

protected:
	QTextEdit* _text_edit;
};

}

#endif
