#include <pvkernel/core/general.h>
#include <picviz/PVView.h>

#include <PVPlainTextEditor.h>

#include <QPushButton>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QStringBuilder>
#include <QMessageBox>

#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/rush/PVUnicodeSource.h>
#include <pvkernel/rush/PVInputFile.h>

/******************************************************************************
 *
 * PVCore::PVPlainTextEditor::PVPlainTextEditor
 *
 *****************************************************************************/
PVInspector::PVPlainTextEditor::PVPlainTextEditor(Picviz::PVView& view, QWidget *parent):
	QWidget(parent),
	_view(view)
{
	_text_edit = new QPlainTextEdit();
	QFontMetrics m(_text_edit->font());
	_text_edit->setMinimumHeight(5*m.lineSpacing());
	_text_edit->installEventFilter(this);
	_text_edit->setSizePolicy(QSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding));
	QPushButton* import_file = new QPushButton(tr("Import from file..."));

	QHBoxLayout* layout = new QHBoxLayout();
	layout->addWidget(_text_edit);
	layout->addWidget(import_file);

	setLayout(layout);
	setFocusPolicy(Qt::WheelFocus);

	connect(import_file, SIGNAL(clicked()), this, SLOT(slot_import_file()));
}

void PVInspector::PVPlainTextEditor::slot_import_file()
{
	QString file = QFileDialog::getOpenFileName(this, tr("Select a text file..."));
	if (file.isEmpty()) {
		return;
	}

	// Use PVUnicodeSource to read the text file. It gives us automatic charset detection !
	try {
		PVFilter::PVChunkFilter null_filter;
		PVRush::PVInput_p ifile(new PVRush::PVInputFile(qPrintable(file)));
		// 100Ko of chunk size
		PVRush::PVUnicodeSource<std::allocator> txt_src(ifile, 1024*100, null_filter.f());
		PVCore::PVChunk* chunk = txt_src();
		QString txt;
		while (chunk) {
			PVCore::list_elts const& elts = chunk->c_elements();
			PVCore::list_elts::const_iterator it;
			for (it = elts.begin(); it != elts.end(); it++) {
				txt = txt % (*it)->get_qstr() % QString("\n");
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

PVCore::PVPlainTextType PVInspector::PVPlainTextEditor::get_text() const
{
	PVCore::PVPlainTextType ret;
	ret.set_text(_text_edit->toPlainText());
	return ret;
}

void PVInspector::PVPlainTextEditor::set_text(PVCore::PVPlainTextType const& text)
{
	_text_edit->setPlainText(text.get_text());
}

bool PVInspector::PVPlainTextEditor::eventFilter(QObject* object, QEvent* event)
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
