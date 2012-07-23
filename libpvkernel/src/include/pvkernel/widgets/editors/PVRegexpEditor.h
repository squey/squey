/**
 * \file PVRegexpEditor.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVCORE_PVREGEXPEDITOR_H
#define PVCORE_PVREGEXPEDITOR_H

#include <QLineEdit>
#include <QRegExp>

#include <pvkernel/core/general.h>

namespace PVWidgets {

/**
 * \class PVRegexpEditor
 */
class PVRegexpEditor : public QLineEdit
{
	Q_OBJECT
	Q_PROPERTY(QRegExp _rx READ get_rx WRITE set_rx USER true)

public:
	PVRegexpEditor(QWidget *parent = 0);
	virtual ~PVRegexpEditor();

public:
	QRegExp get_rx() const;
	void set_rx(QRegExp rx);
};

}

#endif
