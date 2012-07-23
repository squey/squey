/**
 * \file PVEnumEditor.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVEnumType.h>

#include <pvkernel/widgets/editors/PVEnumEditor.h>

/******************************************************************************
 *
 * PVCore::PVEnumEditor::PVEnumEditor
 *
 *****************************************************************************/
PVWidgets::PVEnumEditor::PVEnumEditor(QWidget *parent):
	QComboBox(parent)
{
}

/******************************************************************************
 *
 * PVWidgets::PVEnumEditor::~PVEnumEditor
 *
 *****************************************************************************/
PVWidgets::PVEnumEditor::~PVEnumEditor()
{
}

/******************************************************************************
 *
 * PVWidgets::PVEnumEditor::set_enum
 *
 *****************************************************************************/
void PVWidgets::PVEnumEditor::set_enum(PVCore::PVEnumType e)
{
	_e = e;
	clear();
	addItems(e.get_list());
	setCurrentIndex(e.get_sel_index());
}

PVCore::PVEnumType PVWidgets::PVEnumEditor::get_enum() const
{
	PVCore::PVEnumType ret(_e);
	ret.set_sel(currentIndex());
	return ret;
}
