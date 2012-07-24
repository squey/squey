/**
 * \file PVTextEditEditor.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/general.h>
#include <picviz/PVView.h>

#include <PVTextEditEditor.h>

#include <QPushButton>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QTextEdit>

#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/rush/PVChunkAlign.h>
#include <pvkernel/rush/PVChunkTransformUTF16.h>
#include <pvkernel/rush/PVRawSource.h>
#include <pvkernel/rush/PVInputFile.h>

/******************************************************************************
 *
 * PVCore::PVTextEditEditor::PVTextEditEditor
 *
 *****************************************************************************/
PVInspector::PVTextEditEditor::PVTextEditEditor(Picviz::PVView& view, QWidget *parent):
	QWidget(parent),
	_view(view)
{
	_text_edit = new QTextEdit();
	QFontMetrics m(_text_edit->font());
	// _text_edit->installEventFilter(this);

	QHBoxLayout* layout = new QHBoxLayout();
	layout->addWidget(_text_edit);

	setLayout(layout);
	setFocusPolicy(Qt::WheelFocus);
}


PVCore::PVTextEditType PVInspector::PVTextEditEditor::get_text() const
{
	PVCore::PVTextEditType ret;
	ret.set_text(_text_edit->toTextEdit());
	return ret;
}

void PVInspector::PVTextEditEditor::set_text(PVCore::PVTextEditType const& text)
{
	_text_edit->setTextEdit(text.get_text());
}

// bool PVInspector::PVTextEditEditor::eventFilter(QObject* object, QEvent* event)
// {
// 	if (event->type() == QEvent::FocusOut)
// 	{
// 		if (object == (QObject*) _text_edit) {
// 			// AG: force the widget to lose focus
// 			// Using setFocusProxy with _text_edit does not seem to work...
// 			// Same as in PVTimeFormatEditor.cpp
// 			setFocus(Qt::MouseFocusReason);
// 			clearFocus();
// 		}
// 	}
// 	return QWidget::eventFilter(object, event);
// }
