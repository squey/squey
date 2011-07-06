//! \file PVEnumEditor.h
//! $Id: PVEnumEditor.h 2498 2011-04-25 14:27:23Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVCORE_PVENUMEDITOR_H
#define PVCORE_PVENUMEDITOR_H

#include <QComboBox>
#include <QString>
#include <QWidget>

#include <pvcore/general.h>
#include <pvcore/PVEnumType.h>

#include <picviz/PVView.h>

namespace PVInspector {

/**
 * \class PVEnumEditor
 */
class LibExport PVEnumEditor : public QComboBox
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVEnumType _enum READ get_enum WRITE set_enum USER true)

public:
	PVEnumEditor(Picviz::PVView& view, QWidget *parent = 0);
	virtual ~PVEnumEditor();

public:
	PVCore::PVEnumType get_enum() const;
	void set_enum(PVCore::PVEnumType e);

protected:
	Picviz::PVView& _view;
	PVCore::PVEnumType _e;
};

}

#endif // PVCORE_PVEnumEDITOR_H
