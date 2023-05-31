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

#include <pvkernel/core/PVProgressBox.h>
#include <pvhwloc.h>
#include <pvkernel/core/PVUtils.h>

#include <pvkernel/rush/PVNraw.h>

#include <pvguiqt/PVListDisplayDlg.h>

#include <QClipboard>
#include <QMessageBox>
#include <QMenu>
#include <QHeaderView>

#include <pvkernel/core/PVLogger.h>
#include <pvkernel/core/PVAlgorithms.h>
#include <pvkernel/core/squey_bench.h>
#include <pvkernel/widgets/PVFileDialog.h>

#include <tbb/blocked_range.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_reduce.h>

#include <boost/thread.hpp>

PVGuiQt::PVListDisplayDlg::PVListDisplayDlg(PVAbstractTableModel* model, QWidget* parent)
    : QDialog(parent), _model(model)
{
	assert(_model->parent() == nullptr && "Model should not have parent as we destroy it");
	setupUi(this);

	setAttribute(Qt::WA_DeleteOnClose);

	// Define default field separator
	_field_separator_button->setClearButtonShow(PVWidgets::QKeySequenceWidget::NoShow);
	_field_separator_button->setKeySequence(QKeySequence(Qt::Key_Comma));
	_field_separator_button->setMaxNumKey(1);

	// Define default line separator
	_line_separator_button->setClearButtonShow(PVWidgets::QKeySequenceWidget::NoShow);
	_line_separator_button->setKeySequence(QKeySequence(Qt::Key_Return));
	_line_separator_button->setMaxNumKey(1);

	// `_values_view' is a PVAbstractTableView to handle huge number of values.
	_values_view->setModel(model);
	_values_view->setGridStyle(Qt::NoPen);

	_values_view->horizontalHeader()->setStretchLastSection(true);
	_values_view->horizontalHeader()->setMinimumSectionSize(50);

	_values_view->verticalHeader()->setDefaultSectionSize(
	    _values_view->verticalHeader()->minimumSectionSize());
	_values_view->verticalHeader()->setSectionResizeMode(QHeaderView::Fixed);

	// Define context menu to copy values in clipboard.
	_copy_values_act = new QAction(tr("Copy values"), this);

	_ctxt_menu = new QMenu(this);
	_ctxt_menu->addAction(_copy_values_act);

	_nb_values_edit->setText(QString("%L1").arg(model->size()));

	set_description(QString());

	// Show contextual menu on right click in the table (set menuPolicy to emit
	// signals)
	connect(_values_view, &QWidget::customContextMenuRequested, this,
	        &PVListDisplayDlg::show_ctxt_menu);
	_values_view->setContextMenuPolicy(Qt::CustomContextMenu);

	connect(_btn_copy_clipboard, &QAbstractButton::clicked, this,
	        &PVListDisplayDlg::copy_all_to_clipboard);
	connect(_btn_copy_file, &QAbstractButton::clicked, this, &PVListDisplayDlg::copy_to_file);
	connect(_btn_append_file, &QAbstractButton::clicked, this, &PVListDisplayDlg::append_to_file);
}

PVGuiQt::PVListDisplayDlg::~PVListDisplayDlg()
{
	// Force deletion so that the internal array is destroyed!
	delete _model;
}

void PVGuiQt::PVListDisplayDlg::show_ctxt_menu(const QPoint& pos)
{
	_values_view->table_model()->commit_selection();

	QModelIndex index = _values_view->indexAt(pos);

	if (not index.isValid()) {
		// no contextual menu outside of valid rows
		return;
	}

	// Show the menu at the given pos
	QAction* act_sel = _ctxt_menu->exec(QCursor::pos());

	process_context_menu(act_sel);
}

bool PVGuiQt::PVListDisplayDlg::process_context_menu(QAction* act)
{
	// act can be null if user click outsize the context_menu
	if (act and act == _copy_values_act) {
		copy_selected_to_clipboard();
		return true;
	}
	return false;
}

void PVGuiQt::PVListDisplayDlg::copy_all_to_clipboard()
{
	ask_for_copying_count();

	QString content;

	// TODO : Why we don't check return value? Exception would have avoid it :-)
	export_values(model().size(), content);

	QApplication::clipboard()->setText(content);
}

void PVGuiQt::PVListDisplayDlg::copy_selected_to_clipboard()
{
	QApplication::setOverrideCursor(Qt::BusyCursor);

	// Get the field separator to use for export (defined in UI)
	QString fsep(QChar::fromLatin1(PVWidgets::QKeySequenceWidget::get_ascii_from_sequence(
	    _field_separator_button->keySequence())));
	if (fsep.isEmpty()) {
		fsep = ",";
	}

	// Get the line separator to use for export (defined in UI)
	QString lsep(QChar::fromLatin1(PVWidgets::QKeySequenceWidget::get_ascii_from_sequence(
	    _line_separator_button->keySequence())));
	if (lsep.isEmpty()) {
		lsep = "\n";
	}

	QString content;

	PVCore::PVProgressBox::progress(
	    [&](PVCore::PVProgressBox& /*pbox*/) {
		    /* the PVSelection can be safely traversed because the
		     *range selection
		     * has been committed earlier at right click.
		     *
		     * Manual iteration is done to keep sort ordering.
		     */
		    for (size_t row : model().shown_lines()) {
			    if (not _model->current_selection().get_line(row)) {
				    continue;
			    }
			    boost::this_thread::interruption_point();

			    QString s = model().export_line(row, fsep);
			    if (!s.isNull()) {
				    content.append(s.append(lsep));
			    }
		    }
		},
	    QObject::tr("Copying values..."), this);

	QApplication::clipboard()->setText(content);

	QApplication::restoreOverrideCursor();
}

void PVGuiQt::PVListDisplayDlg::export_to_file(QFile& file)
{
	QString path(file.fileName());
	QTextStream outstream(&file);

	QString content;
	bool success = export_values(model().size(), content);

	outstream << content;

	if (!success) {
		if (QMessageBox::question(this, tr("File writing cancelled"),
		                          tr("Do you want to remove the file '%1'?").arg(path),
		                          QMessageBox::Yes | QMessageBox::No,
		                          QMessageBox::Yes) == QMessageBox::Yes) {
			if (!file.remove()) {
				QMessageBox::warning(
				    this, tr("File writing cancelled"),
				    tr("Error while removing '%1': %2").arg(path).arg(file.errorString()));
			}
		}
		return;
	}
}

bool PVGuiQt::PVListDisplayDlg::export_values(int count, QString& content)
{
	QApplication::setOverrideCursor(Qt::BusyCursor);

	// Get the field separator to use for export (defined in UI)
	QString fsep(QChar::fromLatin1(PVWidgets::QKeySequenceWidget::get_ascii_from_sequence(
	    _field_separator_button->keySequence())));
	if (fsep.isEmpty()) {
		fsep = ",";
	}

	// Get the line separator to use for export (defined in UI)
	QString lsep(QChar::fromLatin1(PVWidgets::QKeySequenceWidget::get_ascii_from_sequence(
	    _line_separator_button->keySequence())));
	if (lsep.isEmpty()) {
		lsep = "\n";
	}

	// Define parallel execution environment
	const size_t nthreads = pvhwloc::core_count();
	tbb::task_scheduler_init init(nthreads);

	auto res = PVCore::PVProgressBox::progress(
	    [&, count](PVCore::PVProgressBox& /*pbox*/) {
		    tbb::task_group_context ctxt;
		    try {

			    BENCH_START(export_values);
			    content = tbb::parallel_reduce(
			        tbb::blocked_range<int>(0, count, std::max(nthreads, count / nthreads)),
			        content,
			        [&](const tbb::blocked_range<int>& range, QString l) -> QString {
				        for (int i = range.begin(); i != range.end(); ++i) {
					        boost::this_thread::interruption_point();
					        QString s = _model->export_line(model().row_pos_to_index(i), fsep);
					        if (!s.isNull()) {
						        l.append(s.append(lsep));
					        }
				        }
				        return l;
				    },
			        // Get ordered result
			        [](const QString& left, const QString& right) -> QString {
				        const_cast<QString&>(left).append(
				            right); // const_cast needed to use optimized append method
				        return left;
				    },
			        tbb::simple_partitioner(), ctxt);
			    BENCH_END(export_values, "export_values", 0, 0, 1, content.size());
		    } catch (boost::thread_interrupted) {
			    ctxt.cancel_group_execution();
			    throw;
		    }
		},
	    QObject::tr("Copying values..."), this);

	QApplication::restoreOverrideCursor();

	return res == PVCore::PVProgressBox::CancelState::CONTINUE;
}

void PVGuiQt::PVListDisplayDlg::export_to_file_ui(bool append)
{
	ask_for_copying_count();

	QFileDialog::Options options;
	if (append) {
		options = QFileDialog::DontConfirmOverwrite;
	}
	QString path = PVWidgets::PVFileDialog::getSaveFileName(this, tr("Save to file..."), QString(),
	                                                        QString(), nullptr, options);
	if (path.isEmpty()) {
		return;
	}

	QFile file(path);
	QIODevice::OpenMode mode = QIODevice::WriteOnly | QIODevice::Text;
	if (append) {
		mode |= QIODevice::Append;
	} else {
		mode |= QIODevice::Truncate;
	}
	if (!file.open(mode)) {
		QMessageBox::critical(
		    this, tr("Copy to file..."),
		    tr("Unable to open '%1' for writing: %2.").arg(path).arg(file.errorString()));
		return;
	}
	export_to_file(file);
}

void PVGuiQt::PVListDisplayDlg::set_description(QString const& desc)
{
	if (desc.isEmpty()) {
		_desc_label->setVisible(false);
	} else {
		_desc_label->setVisible(true);
		_desc_label->setText(desc);
	}
}
