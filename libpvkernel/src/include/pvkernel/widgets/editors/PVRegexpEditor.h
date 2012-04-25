//! \file PVAxisIndexEditor.h
//! $Id: PVAxisIndexEditor.h 2498 2011-04-25 14:27:23Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

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
