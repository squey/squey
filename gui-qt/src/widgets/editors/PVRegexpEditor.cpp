/**
 * \file PVRegexpEditor.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <pvkernel/core/general.h>
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
