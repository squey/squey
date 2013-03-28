/**
 * \file PVPlainTextEditor.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PVPLAINTEXTEDITOR_H
#define PVCORE_PVPLAINTEXTEDITOR_H

#include <QFileDialog>
#include <QWidget>
#include <QPlainTextEdit>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVPlainTextType.h>

namespace PVWidgets {

/**
 * \class PVRegexpEditor
 */
class PVPlainTextEditor: public QWidget
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVPlainTextType _text READ get_text WRITE set_text USER true)

public:
	PVPlainTextEditor(QWidget *parent = 0);

public:
	PVCore::PVPlainTextType get_text() const;
	void set_text(PVCore::PVPlainTextType const& text);

protected:
	virtual bool eventFilter(QObject *object, QEvent *event);
	void save_to_file(bool append);

protected slots:
	void slot_import_file();
	void slot_export_file();
	void slot_export_and_import_file();

protected:
	QPlainTextEdit* _text_edit;

private:
	QFileDialog _file_dlg;
};

}

#endif
