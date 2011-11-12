//! \file PVCheckBoxEditor.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVCORE_PVCHECKBOXEDITOR_H
#define PVCORE_PVCHECKBOXEDITOR_H

#include <QCheckBox>
#include <QString>
#include <QWidget>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVCheckBoxType.h>

#include <picviz/PVView.h>

namespace PVInspector {

/**
 * \class PVCheckBoxEditor
 */
class PVCheckBoxEditor : public QCheckBox
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVCheckBoxType _checked READ get_checked WRITE set_checked USER true)

public:
	PVCheckBoxEditor(Picviz::PVView& view, QWidget *parent = 0);
	virtual ~PVCheckBoxEditor();

public:
	PVCore::PVCheckBoxType get_checked() const;
	void set_checked(PVCore::PVCheckBoxType e);

protected:
	Picviz::PVView& _view;
	PVCore::PVCheckBoxType _e;
};

}

#endif // PVCORE_PVCheckBoxEDITOR_H
