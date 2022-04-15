//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/widgets/editors/PVPlainTextEditor.h>

#include <QPushButton>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QTextStream>
#include <QTextCodec>

#include <pvkernel/core/PVConfig.h>
#include <pvkernel/core/PVTextChunk.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/rush/PVUnicodeSource.h>
#include <pvkernel/rush/PVInputFile.h>

/******************************************************************************
 *
 * PVCore::PVPlainTextEditor::PVPlainTextEditor
 *
 *****************************************************************************/
PVWidgets::PVPlainTextEditor::PVPlainTextEditor(QWidget* parent) : QWidget(parent)
{
	_file_dlg.setDirectory(PVCore::PVConfig::get().get_lists_dir());

	_text_edit = new QPlainTextEdit();
	QFontMetrics m(_text_edit->font());
	_text_edit->setMinimumHeight(5 * m.lineSpacing());
	_text_edit->installEventFilter(this);
	_text_edit->setSizePolicy(
	    QSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding));
	auto import_file = new QPushButton();
	import_file->setIcon(QIcon(":/import_file"));
	import_file->setIconSize(QSize(24, 24));
	import_file->setToolTip(tr("Import from file..."));
	import_file->setFlat(true);
	auto export_file = new QPushButton();
	export_file->setIcon(QIcon(":/export_file"));
	export_file->setIconSize(QSize(24, 24));
	export_file->setToolTip(tr("Export to file..."));
	export_file->setFlat(true);
	auto export_and_append_file = new QPushButton();
	export_and_append_file->setIcon(QIcon(":/append_file"));
	export_and_append_file->setIconSize(QSize(24, 24));
	export_and_append_file->setToolTip(tr("Export and append to file..."));
	export_and_append_file->setFlat(true);

	auto layout = new QHBoxLayout();
	layout->setContentsMargins(0, 0, 0, 0);
	layout->addWidget(_text_edit);
	auto files_layer = new QVBoxLayout();
	layout->addLayout(files_layer);
	files_layer->addWidget(import_file);
	files_layer->addWidget(export_file);
	files_layer->addWidget(export_and_append_file);

	setLayout(layout);
	setFocusPolicy(Qt::WheelFocus);

	connect(import_file, &QAbstractButton::clicked, this, &PVPlainTextEditor::slot_import_file);
	connect(export_file, &QAbstractButton::clicked, this, &PVPlainTextEditor::slot_export_file);
	connect(export_and_append_file, &QAbstractButton::clicked, this,
	        &PVPlainTextEditor::slot_export_and_import_file);
}

void PVWidgets::PVPlainTextEditor::save_to_file(const bool append)
{
	QString file = _file_dlg.getSaveFileName(this, tr("Save text file..."),
	                                         _file_dlg.directory().absolutePath());

	QFile outfile;
	outfile.setFileName(file);
	QIODevice::OpenMode flags = QIODevice::Text | QIODevice::WriteOnly;
	if (append) {
		flags |= QIODevice::ReadOnly;
	}
	outfile.open(flags);

	const size_t file_size = outfile.size();
	if (append && file_size > 0) {
		outfile.seek(file_size - 1);
		char last_char = 0;
		outfile.read(&last_char, 1);
		if (last_char != '\n') {
			last_char = '\n';
			outfile.write(&last_char, 1);
		}
	}

	QTextStream out(&outfile);
	out.setEncoding(QStringConverter::Utf8); // by default anyway
	QString text_write = _text_edit->toPlainText();
	if (*(text_write.constEnd() - 1) != QChar('\n')) {
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
	QString file = _file_dlg.getOpenFileName(this, tr("Open text file..."),
	                                         _file_dlg.directory().absolutePath());
	if (file.isEmpty()) {
		return;
	}

	// Use PVUnicodeSource to read the text file. It gives us automatic charset detection !
	try {
		PVRush::PVInputFile* pfile = new PVRush::PVInputFile(qPrintable(file));
		PVRush::PVInput_p input(pfile);
		PVRush::PVUnicodeSource<std::allocator> txt_src(input, 10 * 1024 * 1024);
		PVCore::PVTextChunk* chunk = txt_src();
		std::string txt;
		while (chunk) {
			for (auto const* elt : chunk->c_elements()) {
				txt += std::string(elt->begin(), elt->size()) + "\n";
			}
			chunk->free();
			chunk = txt_src();
		}
		// Remove last carriage return if otherwise we would search for empty strings as well
		txt.pop_back();
		_text_edit->setPlainText(QString::fromStdString(txt));

	} catch (PVRush::PVInputException const& ex) {
		QMessageBox* box =
		    new QMessageBox(QMessageBox::Critical, tr("Error while opening file..."),
		                    QString::fromStdString(ex.what()), QMessageBox::Ok, this);
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
	if (event->type() == QEvent::FocusOut) {
		if (object == (QObject*)_text_edit) {
			// AG: force the widget to lose focus
			// Using setFocusProxy with _text_edit does not seem to work...
			// Same as in PVTimeFormatEditor.cpp
			setFocus(Qt::MouseFocusReason);
			clearFocus();
		}
	}
	return QWidget::eventFilter(object, event);
}
