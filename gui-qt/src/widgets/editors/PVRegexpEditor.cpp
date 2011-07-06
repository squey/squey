//! \file PVRegexpEditor.cpp
//! $Id: PVRegexpEditor.cpp 2699 2011-05-12 03:58:48Z cdash $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvcore/general.h>
#include <picviz/PVView.h>

#include <PVRegexpEditor.h>

/******************************************************************************
 *
 * PVCore::PVRegexpEditor::PVRegexpEditor
 *
 *****************************************************************************/
PVInspector::PVRegexpEditor::PVRegexpEditor(Picviz::PVView& view, QWidget *parent):
	QLineEdit(parent),
	_view(view)
{
}

/******************************************************************************
 *
 * PVInspector::PVRegexpEditor::~PVRegexpEditor
 *
 *****************************************************************************/
PVInspector::PVRegexpEditor::~PVRegexpEditor()
{
}

void PVInspector::PVRegexpEditor::set_rx(QRegExp rx)
{
	setText(rx.pattern());
}

QRegExp PVInspector::PVRegexpEditor::get_rx() const
{
	return QRegExp(text());
}
