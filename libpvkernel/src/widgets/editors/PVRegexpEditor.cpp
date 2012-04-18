//! \file PVRegexpEditor.cpp
//! $Id: PVRegexpEditor.cpp 2699 2011-05-12 03:58:48Z cdash $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvkernel/core/general.h>
#include <pvkernel/widgets/editors/PVRegexpEditor.h>

/******************************************************************************
 *
 * PVCore::PVRegexpEditor::PVRegexpEditor
 *
 *****************************************************************************/
PVWidgets::PVRegexpEditor::PVRegexpEditor(QWidget *parent):
	QLineEdit(parent)
{
}

/******************************************************************************
 *
 * PVWidgets::PVRegexpEditor::~PVRegexpEditor
 *
 *****************************************************************************/
PVWidgets::PVRegexpEditor::~PVRegexpEditor()
{
}

void PVWidgets::PVRegexpEditor::set_rx(QRegExp rx)
{
	setText(rx.pattern());
}

QRegExp PVWidgets::PVRegexpEditor::get_rx() const
{
	return QRegExp(text());
}
