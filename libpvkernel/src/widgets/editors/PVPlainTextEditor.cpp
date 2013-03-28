/**
 * \file PVPlainTextEditor.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/general.h>
#include <pvkernel/widgets/editors/PVPlainTextEditor.h>

#include <QPushButton>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QTextStream>

#include <pvkernel/core/PVConfig.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/rush/PVChunkAlign.h>
#include <pvkernel/rush/PVChunkTransformUTF16.h>
#include <pvkernel/rush/PVRawSource.h>
#include <pvkernel/rush/PVInputFile.h>

/******************************************************************************
 *
 * PVCore::PVPlainTextEditor::PVPlainTextEditor
 *
 *****************************************************************************/
PVWidgets::PVPlainTextEditor::PVPlainTextEditor(QWidget *parent):
	QWidget(parent)
{
	_file_dlg.setDirectory(PVCore::PVConfig::get().get_lists_dir());

	_text_edit = new QPlainTextEdit();
	QFontMetrics m(_text_edit->font());
	_text_edit->setMinimumHeight(5*m.lineSpacing());
	_text_edit->installEventFilter(this);
	_text_edit->setSizePolicy(QSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding));
	QPushButton* import_file = new QPushButton();
	import_file->setIcon(QIcon(":/import_file"));
	import_file->setIconSize(QSize(24, 24));
	import_file->setToolTip(tr("Import from file..."));
	import_file->setFlat(true);
	QPushButton* export_file = new QPushButton();
	export_file->setIcon(QIcon(":/export_file"));
	export_file->setIconSize(QSize(24, 24));
	export_file->setToolTip(tr("Export to file..."));
	export_file->setFlat(true);
	QPushButton* export_and_append_file = new QPushButton();
	export_and_append_file->setIcon(QIcon(":/append_file"));
	export_and_append_file->setIconSize(QSize(24, 24));
	export_and_append_file->setToolTip(tr("Export and append to file..."));
	export_and_append_file->setFlat(true);

	QHBoxLayout* layout = new QHBoxLayout();
	layout->addWidget(_text_edit);
	QVBoxLayout* files_layer = new QVBoxLayout();
	layout->addLayout(files_layer);
	files_layer->addWidget(import_file);
	files_layer->addWidget(export_file);
	files_layer->addWidget(export_and_append_file);

	setLayout(layout);
	setFocusPolicy(Qt::WheelFocus);

	connect(import_file, SIGNAL(clicked()), this, SLOT(slot_import_file()));
	connect(export_file, SIGNAL(clicked()), this, SLOT(slot_export_file()));
	connect(export_and_append_file, SIGNAL(clicked()), this, SLOT(slot_export_and_import_file()));
}

void PVWidgets::PVPlainTextEditor::save_to_file(const bool append)
{
	QString file = _file_dlg.getSaveFileName(this, tr("Save text file..."), _file_dlg.directory().absolutePath());

    QFile outfile;
    outfile.setFileName(file);
	QIODevice::OpenMode flags = QIODevice::Text | QIODevice::WriteOnly;
	if (append) {
		flags |= QIODevice::ReadOnly;
	}
    outfile.open(flags);

	const size_t file_size = outfile.size();
	if (append && file_size > 0) {
		outfile.seek(file_size-1);
		char last_char = 0;
		outfile.read(&last_char, 1);
		if (last_char != '\n') {
			last_char = '\n';
			outfile.write(&last_char, 1);
		}
	}

    QTextStream out(&outfile);
	out.setCodec(QTextCodec::codecForName("UTF-8"));
	QString text_write = _text_edit->toPlainText();
	if (*(text_write.constEnd()-1) != QChar('\n')) {
		text_write.append(QChar('\n'));
	}
    out << text_write;
}

void PVWidgets::PVPlainTextEditor::slot_export_file()
{
	save_to_file(false);
}

void PVWidgets::PVPlainTextEditor::slot_export_and_import_file()
{
	save_to_file(true);
}

void PVWidgets::PVPlainTextEditor::slot_import_file()
{
	QString file = _file_dlg.getOpenFileName(this, tr("Open text file..."), _file_dlg.directory().absolutePath());
	if (file.isEmpty()) {
		return;
	}

	// Use PVUnicodeSource to read the text file. It gives us automatic charset detection !
	try {
		PVFilter::PVChunkFilter null_filter;
		PVRush::PVChunkTransformUTF16 trans_utf16;
		PVRush::PVChunkAlign null_align;
		PVRush::PVInputFile* pfile = new PVRush::PVInputFile(qPrintable(file));
		PVRush::PVInput_p input(pfile);
		PVRush::PVRawSource<std::allocator> txt_src(input, null_align, 10*1024*1024, trans_utf16, null_filter.f());
		PVCore::PVChunk* chunk = txt_src();
		QString txt("");
		QString str_tmp;
		while (chunk) {
			PVCore::list_elts const& elts = chunk->c_elements();
			PVCore::list_elts::const_iterator it;
			for (it = elts.begin(); it != elts.end(); it++) {
				txt += (*it)->get_qstr(str_tmp);
			}
			chunk->free();
			chunk = txt_src();
		}
		_text_edit->setPlainText(txt);
	}
	catch (PVRush::PVInputException const& ex) {
		QMessageBox* box = new QMessageBox(QMessageBox::Critical, tr("Error while opening file..."), QString::fromStdString(ex.what()), QMessageBox::Ok, this);
		box->exec();
	}
}

PVCore::PVPlainTextType PVWidgets::PVPlainTextEditor::get_text() const
{
	PVCore::PVPlainTextType ret;
	ret.set_text(_text_edit->toPlainText());
	return ret;
}

void PVWidgets::PVPlainTextEditor::set_text(PVCore::PVPlainTextType const& text)
{
	_text_edit->setPlainText(text.get_text());
}

bool PVWidgets::PVPlainTextEditor::eventFilter(QObject* object, QEvent* event)
{
	if (event->type() == QEvent::FocusOut)
	{
		if (object == (QObject*) _text_edit) {
			// AG: force the widget to lose focus
			// Using setFocusProxy with _text_edit does not seem to work...
			// Same as in PVTimeFormatEditor.cpp
			setFocus(Qt::MouseFocusReason);
			clearFocus();
		}
	}
	return QWidget::eventFilter(object, event);
}
