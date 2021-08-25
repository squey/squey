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

#include <QTableView>
#include <QProgressDialog>

#include "include/TreeWidget.h"
#include "ui_TreeWidget.h"

#include "rapidjson/document.h"

#include <profile-ui/models/include/ProtocolFieldListModel.h>

#include <pvkernel/widgets/PVFileDialog.h>

TreeWidget::TreeWidget(rapidjson::Document* json_data, QWidget* parent)
    : QWidget(parent)
    , _ui(new Ui::TreeWidget)
    , _tree_model(new pvpcap::PcapTreeModel(json_data, false, this))
{
	_ui->setupUi(this);

	_ui->tree_view->setModel(_tree_model);

	_ui->field_view->setModel(nullptr);

	// stretch last columns
	_ui->field_view->horizontalHeader()->setStretchLastSection(true);
	_ui->tree_view->header()->setStretchLastSection(true);

	// update field model only when selection changes, but update currently selected field also
	connect(_ui->tree_view, &QTreeView::clicked, this,
	        &TreeWidget::update_field_model_with_selected_field);
	connect(_ui->tree_view->selectionModel(), &QItemSelectionModel::selectionChanged,
	        [&](const QItemSelection& selected, const QItemSelection&) {
		        update_field_model(selected.indexes()[0]);
		    });
}

TreeWidget::~TreeWidget()
{
	// delete models if it exists
	delete _ui->field_view->model();
	delete _ui->field_view->selectionModel();
	delete _ui->tree_view->model();
	delete _ui->tree_view->selectionModel();

	delete _ui;
}

void TreeWidget::on_select_button_clicked()
{
	QString filename = PVWidgets::PVFileDialog::getOpenFileName(this, tr("Open PCAP file"), "",
	                                                            tr("PCAP file (*.pcap *.pcapng)"));

	if (not filename.isEmpty()) {
		_ui->select_edit->setText(filename);

		// re-initialise protocol fields detail view
		// Remove old model. We can use Qt for this with this widget as parent but
		// they will be deleted at the end which will use memory too long.
		// We have to handle the selectionModel too.
		QAbstractItemModel* m = _ui->field_view->model();
		QItemSelectionModel* sm = _ui->field_view->selectionModel();
		if (m) {
			m->disconnect();
		}
		delete sm;
		delete m;
		_ui->field_view->setModel(nullptr);

		QProgressDialog* loading_progress_dialog =
		    new QProgressDialog("Extracting protocols", "Cancel", 0, 0, this);
		loading_progress_dialog->setWindowModality(Qt::WindowModal);

		QThread* loading_thread = new QThread;
		ExtractProtocolsWorker* worker = new ExtractProtocolsWorker(_tree_model, filename);
		worker->moveToThread(loading_thread);

		connect(loading_thread, &QThread::started, worker, &ExtractProtocolsWorker::load);
		connect(worker, &ExtractProtocolsWorker::finished, [=,this]() {
			loading_progress_dialog->close();
			loading_progress_dialog->deleteLater();

			Q_EMIT update_tree_data();

			_ui->tree_view->expandAll();
			_ui->tree_view->resizeColumnToContents(0);

			loading_thread->quit();
			loading_thread->deleteLater();
		});
		connect(loading_progress_dialog, &QProgressDialog::canceled, worker,
		        &ExtractProtocolsWorker::cancel);

		loading_progress_dialog->show();
		loading_thread->start();
	}
}

void TreeWidget::update_field_model_with_selected_field(const QModelIndex& index)
{
	if (index == _selected_protocol) {
		update_field_model(index);
	}
}

void TreeWidget::update_field_model(const QModelIndex& index)
{
	// Remove old model. We can use Qt for this with this widget as parent but
	// they will be deleted at the end which will use memory too long.
	// We have to handle the selectionModel too.
	QAbstractItemModel* m = _ui->field_view->model();
	QItemSelectionModel* sm = _ui->field_view->selectionModel();

	if (m) {
		m->disconnect();
	}

	// If the index is not valid, just reset the model.
	// Update the model with new selected one if it is an eventId
	if (index.isValid()) {

		pvpcap::JsonTreeItem* index_item =
		    static_cast<pvpcap::JsonTreeItem*>(index.internalPointer());
		rapidjson::Value* fields = &index_item->value()["fields"];
		_ui->field_view->setModel(new pvpcap::ProtocolFieldListModel(fields, this));

		_ui->field_view->resizeColumnToContents(0); // checkbox
		_ui->field_view->resizeColumnToContents(2); // filter name

		connect(static_cast<pvpcap::ProtocolFieldListModel*>(_ui->field_view->model()),
		        &pvpcap::ProtocolFieldListModel::update_selection, this,
		        &TreeWidget::update_selection);

		connect(_ui->field_view, &QTableView::clicked,
		        static_cast<pvpcap::ProtocolFieldListModel*>(_ui->field_view->model()),
		        &pvpcap::ProtocolFieldListModel::select);
	} else {
		_ui->field_view->setModel(nullptr);
	}

	delete sm;
	delete m;

	_selected_protocol = index;
}

void TreeWidget::update_selection()
{
	_ui->tree_view->viewport()->update();

	Q_EMIT propagate_selection();
}

void TreeWidget::update_model()
{
	_tree_model->reset();
	_ui->tree_view->expandAll();
	_ui->tree_view->resizeColumnToContents(0);

	update_field_model(QModelIndex());
}
