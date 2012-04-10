#include <pvkernel/rush/PVNraw.h>
#include <picviz/PVView.h>

#include <pvkernel/core/PVProgressBox.h>
#include <PVListDisplayDlg.h>

#include <QClipboard>
#include <QFileDialog>
#include <QMessageBox>

// PVlistColNrawDlg
//

PVInspector::PVListDisplayDlg::PVListDisplayDlg(QAbstractListModel* model, QWidget* parent):
	QDialog(parent),
	_model(model)
{
	setupUi(this);
	_field_separator_btn->setClearButtonShow(PVWidgets::QKeySequenceWidget::NoShow);
	_field_separator_btn->setKeySequence(QKeySequence(Qt::Key_Return));
	_field_separator_btn->setMaxNumKey(1);

	_values_view->setUniformItemSizes(true);
	_values_view->setModel(_model);
	_values_view->setContextMenuPolicy(Qt::ActionsContextMenu);

	QAction* copy_values_act = new QAction(tr("Copy value in the clipboard..."), this);
	connect(copy_values_act, SIGNAL(triggered()), this, SLOT(copy_value_clipboard()));

	_values_view->addAction(copy_values_act);
	_values_view->setSelectionMode(QAbstractItemView::SingleSelection);

	_nb_values_edit->setText(QString().setNum(model->rowCount()));

	set_description(QString());

	connect(_btn_copy_clipboard, SIGNAL(clicked()), this, SLOT(copy_to_clipboard()));
	connect(_btn_copy_file, SIGNAL(clicked()), this, SLOT(copy_to_file()));
}

bool PVInspector::PVListDisplayDlg::write_values(QDataStream* stream)
{
	QString sep(QChar::fromAscii(PVWidgets::QKeySequenceWidget::get_ascii_from_sequence(_field_separator_btn->keySequence())));
	if (sep.isEmpty()) {
		sep = "\n";
	}
	QByteArray sep_bp = sep.toLocal8Bit();
	QByteArray tmp;
	// Write the values into that data stream.
	// Values are converted into the system locale.
	for (int i = 0; i < _model->rowCount(); i++) {
		if (i % 64 == 0) {
			boost::this_thread::interruption_point();
		}
		tmp = _model->data(_model->index(i, 0)).toString().toLocal8Bit();
		int len = stream->writeRawData(tmp.constData(), tmp.size());
		if (len != tmp.size()) {
			return false;
		}
		len = stream->writeRawData(sep_bp.constData(), sep_bp.size());
		if (len != sep_bp.size()) {
			return false;
		}
	}

	return true;
}

void PVInspector::PVListDisplayDlg::copy_to_clipboard()
{
	// Write the values into a byte array, and copy it into the clipboard
	QByteArray ba;
	QDataStream ds(&ba, QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text);
	bool write_success = false;
	bool process_done = PVCore::PVProgressBox::progress(boost::bind(&PVListDisplayDlg::write_values, this, &ds), tr("Copying values..."), write_success, this);
	if (!process_done) {
		return;
	}
	if (!write_success) {
		QMessageBox::critical(this, tr("Copy to clipboard..."), tr("Error while copying to clipboard: no more memory available."));
		return;
	}

	QClipboard *clipboard = QApplication::clipboard();
	QMimeData* mdata = new QMimeData();
	mdata->setData("text/plain", ba);
	clipboard->setMimeData(mdata);

	QMessageBox::information(this, tr("Copy to clipboard..."), tr("Copy done."));
}

void PVInspector::PVListDisplayDlg::copy_to_file()
{
	QString path = QFileDialog::getSaveFileName(this, tr("Save to file..."));
	if (path.isEmpty()) {
		return;
	}

	QFile file(path);
	if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
		QMessageBox::critical(this, tr("Copy to file..."), tr("Unable to open '%1' for writing: %2.").arg(path).arg(file.errorString()));
		return;
	}

	QDataStream ds(&file);
	bool write_success = false;
	bool process_done = PVCore::PVProgressBox::progress(boost::bind(&PVListDisplayDlg::write_values, this, &ds), tr("Copying values..."), write_success, this);
	if (!process_done) {
		if (QMessageBox::question(this, tr("File writing cancelled"), tr("Do you want to remove the file '%1'?").arg(path), QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes) == QMessageBox::Yes) {
			if (!file.remove()) {
				QMessageBox::warning(this, tr("File writing cancelled"), tr("Error while removing '%1': %2").arg(path).arg(file.errorString()));
			}
		}
		return;
	}
	if (!write_success) {
		QMessageBox::critical(this, tr("Copy to file..."), tr("Error while writing to '%1': %2.").arg(path).arg(file.errorString()));
		return;
	}

	QMessageBox::information(this, tr("Copy to file..."), tr("Copy done."));
}

void PVInspector::PVListDisplayDlg::copy_value_clipboard()
{
	QModelIndex idx = _values_view->currentIndex();
	if (idx.isValid()) {
		QString txt = _model->data(idx).toString();
		QApplication::clipboard()->setText(txt);
	}
}

void PVInspector::PVListDisplayDlg::set_description(QString const& desc)
{
	if (desc.isEmpty()) {
		_desc_label->setVisible(false);
	}
	else {
		_desc_label->setVisible(true);
		_desc_label->setText(desc);
	}
}
