/**
 * \file PVEnumEditor.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVCORE_PVENUMEDITOR_H
#define PVCORE_PVENUMEDITOR_H

#include <QComboBox>
#include <QString>
#include <QWidget>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVEnumType.h>

#include <picviz/PVView.h>

namespace PVInspector {

/**
 * \class PVEnumEditor
 */
class PVEnumEditor : public QComboBox
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
