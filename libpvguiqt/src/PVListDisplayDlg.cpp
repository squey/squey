/**
 * \file PVListDisplayDlg.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/rush/PVNraw.h>

#include <picviz/PVView.h>

#include <pvguiqt/PVListDisplayDlg.h>
#include <pvguiqt/PVStringSortProxyModel.h>

#include <QClipboard>
#include <QMessageBox>
#include <QMenu>

#include <pvkernel/core/PVLogger.h>

#define AUTOMATIC_SORT_MAX_NUMBER 32768

PVGuiQt::PVListDisplayDlg::PVListDisplayDlg(QAbstractListModel* model, QWidget* parent):
	QDialog(parent)
{
	setupUi(this);
	_field_separator_btn->setClearButtonShow(PVWidgets::QKeySequenceWidget::NoShow);
	_field_separator_btn->setKeySequence(QKeySequence(Qt::Key_Return));
	_field_separator_btn->setMaxNumKey(1);

	// Sort proxy model
	PVStringSortProxyModel* proxy_model = new PVStringSortProxyModel(_values_view);
	proxy_model->setSourceModel(model);

	// `_values_view' is a QTableView, because QListView suffers from the same
	// bug than QTableView used to suffer when a "large" (> 75000000) number of
	// items are present in the model. See
	// https://bugreports.qt-project.org/browse/QTBUG-18490 for more
	// informations.
	// The order of the calls here are important, especially the call to
	// setDefaultSectionSize that must be called *before* setModel, or it could
	// take a huge amount of time.

	_values_view->horizontalHeader()->setSortIndicator(0, Qt::AscendingOrder);
	_values_view->horizontalHeader()->setStretchLastSection(true);
	_values_view->verticalHeader()->setDefaultSectionSize(_values_view->verticalHeader()->minimumSectionSize());
	_values_view->setModel(proxy_model);
	_values_view->setGridStyle(Qt::NoPen);
	_values_view->setContextMenuPolicy(Qt::ActionsContextMenu);
	_values_view->verticalHeader()->hide();

	_copy_values_act = new QAction(tr("Copy value in the clipboard..."), this);

	_ctxt_menu = new QMenu(this);
	_hhead_ctxt_menu = new QMenu(this);

	_ctxt_menu->addAction(_copy_values_act);
	_values_view->setSelectionMode(QAbstractItemView::SingleSelection);

	_nb_values_edit->setText(QString().setNum(model->rowCount()));

	set_description(QString());

	connect(this, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(show_ctxt_menu(const QPoint&)));
	setContextMenuPolicy(Qt::CustomContextMenu);

	connect(_btn_copy_clipboard, SIGNAL(clicked()), this, SLOT(copy_to_clipboard()));
	connect(_btn_copy_file, SIGNAL(clicked()), this, SLOT(copy_to_file()));
	connect(_btn_append_file, SIGNAL(clicked()), this, SLOT(append_to_file()));
	connect(_btn_sort, SIGNAL(clicked()), this, SLOT(sort()));

	connect(_values_view->horizontalHeader(), SIGNAL(sectionClicked(int)), this, SLOT(sort_by_column(int)));
	connect(_values_view->horizontalHeader(), SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(show_hhead_ctxt_menu(const QPoint&)));
	_values_view->horizontalHeader()->setContextMenuPolicy(Qt::CustomContextMenu);

	if (model->rowCount() < AUTOMATIC_SORT_MAX_NUMBER) {
		sort();
	}
}

void PVGuiQt::PVListDisplayDlg::show_ctxt_menu(const QPoint& /*pos*/)
{
	// Show the menu at the given pos
	QAction* act_sel = _ctxt_menu->exec(QCursor::pos());

	process_context_menu(act_sel);

}

void PVGuiQt::PVListDisplayDlg::show_hhead_ctxt_menu(const QPoint& pos)
{
	// Show the menu at the given pos
	if (_values_view->horizontalHeader()->logicalIndexAt(pos) > 0) {
		QAction* act_sel = _hhead_ctxt_menu->exec(QCursor::pos());

		process_hhead_context_menu(act_sel);
	}
}

void PVGuiQt::PVListDisplayDlg::process_context_menu(QAction* act)
{
	if (act) {
		if (act == _copy_values_act) {
			copy_value_clipboard();
		}
	}
}

void PVGuiQt::PVListDisplayDlg::process_hhead_context_menu(QAction* act)
{
}

bool PVGuiQt::PVListDisplayDlg::write_values(QDataStream* stream)
{
	QString sep(QChar::fromAscii(PVWidgets::QKeySequenceWidget::get_ascii_from_sequence(_field_separator_btn->keySequence())));
	if (sep.isEmpty()) {
		sep = "\n";
	}
	QByteArray sep_bp = sep.toLocal8Bit();
	QByteArray tmp;
	// Write the values into that data stream.
	// Values are converted into the system locale.
	for (int i = 0; i < model()->rowCount(); i++) {
		if (i % 64 == 0) {
			boost::this_thread::interruption_point();
		}
		tmp = proxy_model()->data(proxy_model()->index(i, 0, QModelIndex())).toString().toLocal8Bit();
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

void PVGuiQt::PVListDisplayDlg::copy_to_clipboard()
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

void PVGuiQt::PVListDisplayDlg::write_to_file_ui(bool append)
{
	QFileDialog::Options options = 0;
	if (append) {
		options = QFileDialog::DontConfirmOverwrite;
	}
	QString path = _file_dlg.getSaveFileName(this, tr("Save to file..."), QString(), QString(), NULL, options);
	if (path.isEmpty()) {
		return;
	}

	QFile file(path);
	QIODevice::OpenMode mode = QIODevice::WriteOnly | QIODevice::Text;
	if (append) {
		mode |= QIODevice::Append;
	}
	else {
		mode |= QIODevice::Truncate;
	}
	if (!file.open(mode)) {
		QMessageBox::critical(this, tr("Copy to file..."), tr("Unable to open '%1' for writing: %2.").arg(path).arg(file.errorString()));
		return;
	}
	write_to_file(file);
}

void PVGuiQt::PVListDisplayDlg::write_to_file(QFile& file)
{
	QString path(file.fileName());
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
}

void PVGuiQt::PVListDisplayDlg::copy_value_clipboard()
{
	QModelIndex idx = _values_view->currentIndex();
	if (idx.isValid()) {
		QString txt = proxy_model()->data(idx).toString();
		QApplication::clipboard()->setText(txt);
	}
}

void PVGuiQt::PVListDisplayDlg::set_description(QString const& desc)
{
	if (desc.isEmpty()) {
		_desc_label->setVisible(false);
	}
	else {
		_desc_label->setVisible(true);
		_desc_label->setText(desc);
	}
}

PVGuiQt::PVStringSortProxyModel* PVGuiQt::PVListDisplayDlg::proxy_model()
{
	return static_cast<PVStringSortProxyModel*>(_values_view->model());
}

QAbstractListModel* PVGuiQt::PVListDisplayDlg::model()
{
	return static_cast<QAbstractListModel*>(proxy_model()->sourceModel());
}

void PVGuiQt::PVListDisplayDlg::sort_by_column(int col)
{
	if (col == 0) {
		sort();
	}
}

void PVGuiQt::PVListDisplayDlg::sort()
{
	Qt::SortOrder order =  _values_view->horizontalHeader()->sortIndicatorOrder();
	proxy_model()->sort(0, order);
	_btn_sort->hide();
}
