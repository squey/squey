//! \file PVEnumEditor.cpp
//! $Id: PVEnumEditor.cpp 2699 2011-05-12 03:58:48Z cdash $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVEnumType.h>

#include <picviz/PVView.h>

#include <PVEnumEditor.h>

/******************************************************************************
 *
 * PVCore::PVEnumEditor::PVEnumEditor
 *
 *****************************************************************************/
PVInspector::PVEnumEditor::PVEnumEditor(Picviz::PVView& view, QWidget *parent):
	QComboBox(parent),
	_view(view)
{
}

/******************************************************************************
 *
 * PVInspector::PVEnumEditor::~PVEnumEditor
 *
 *****************************************************************************/
PVInspector::PVEnumEditor::~PVEnumEditor()
{
}

/******************************************************************************
 *
 * PVInspector::PVEnumEditor::set_enum
 *
 *****************************************************************************/
void PVInspector::PVEnumEditor::set_enum(PVCore::PVEnumType e)
{
	_e = e;
	clear();
	addItems(e.get_list());
	setCurrentIndex(e.get_sel_index());
}

PVCore::PVEnumType PVInspector::PVEnumEditor::get_enum() const
{
	PVCore::PVEnumType ret(_e);
	ret.set_sel(currentIndex());
	return ret;
}
