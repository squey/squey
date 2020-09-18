/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2019
 */

#include "PVERFParamsWidget.h"
#include "PVERFTreeView.h"

#include <pvkernel/core/serialize_numbers.h>
#include <pvkernel/core/PVUtils.h>
#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/widgets/PVFileDialog.h>
#include <pvkernel/widgets/PVUtils.h>
#include <pvkernel/widgets/PVMultipleFileDialog.h>
#include <pvkernel/widgets/PVUtils.h>

#include <QLabel>
#include <QTreeView>
#include <QHBoxLayout>
#include <QSplitter>
#include <QTextEdit>
#include <QPushButton>
#include <QMessageBox>
#include <QGuiApplication>
#include <QScreen>
#include <QDialogButtonBox>
#include <QStackedWidget>
#include <QStatusBar>
#include <QTimer>

#include <pvlogger.h>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

PVRush::PVERFParamsWidget::PVERFParamsWidget(PVInputTypeERF const* /*in_t*/, QWidget* parent)
{
	QScreen* screen = QGuiApplication::primaryScreen();
	QRect screen_geometry = screen->geometry();
	int height = screen_geometry.height();
	int width = screen_geometry.width();

	QSplitter* splitter = new QSplitter(Qt::Horizontal);
	splitter->setFixedSize(QSize(width / 3, height / 2));

	QStatusBar* status_bar = new QStatusBar(this);
	_status_bar_needs_refresh = true;

	_paths = PVWidgets::PVMultipleFileDialog::getOpenFileNames(this, tr("Open ERF file"), "",
	                                                           tr("ERF files (*.erf, *.erfh5)"));

	if (_paths.isEmpty()) {
		setResult(QDialog::Rejected);
		return;
	}

	setResult(QDialog::Accepted);
	_erf.reset(new PVERFAPI(_paths.front().toStdString()));

	_model.reset(new PVRush::PVERFTreeModel(_paths.front()));
	PVRush::PVERFTreeView* tree = new PVRush::PVERFTreeView(_model.get(), parent);
	tree->setSelectionMode(QAbstractItemView::MultiSelection);
	tree->setAlternatingRowColors(true);
	tree->expandAll();

	// Check files structures
	if (_paths.size() > 1) {
		QStringList bad_files;
		PVCore::PVProgressBox::progress(
		    [&](PVCore::PVProgressBox& /*pbox*/) {
			    const rapidjson::Document& ref_doc = _model->save(PVERFTreeModel::ENodesType::ALL);
			    for (int i = 1; i < _paths.size(); i++) {
				    const QString& path = _paths[i];
				    const rapidjson::Document& doc =
				        PVERFTreeModel(path).save(PVERFTreeModel::ENodesType::ALL);
				    if (doc != ref_doc) {
					    bad_files.append(path);
				    }
			    }
		    },
		    "Checking files structures...", this);
		if (not bad_files.empty()) {
			QMessageBox::critical(
			    this, "Incompatible file structure",
			    QString("The following file(s) have not the expected structure :<br><br> %1")
			        .arg(bad_files.join("<br>")),
			    QMessageBox::Ok);
			setResult(QDialog::Rejected);
			return;
		}
	}

	QVBoxLayout* list_layout = new QVBoxLayout;
	QLabel* list_label = new QLabel;

	auto store_list_f = [&](QTextEdit* text_edit) {
		// sender() in lambda function is always returning nullptr
		QModelIndex index = text_edit->property("index").toModelIndex();
		PVERFTreeItem* item = static_cast<PVERFTreeItem*>(index.internalPointer());
		item->set_user_data(text_edit->toPlainText());
		_status_bar_needs_refresh = true;
	};

	QStackedWidget* list_stacked_text = new QStackedWidget;
	QTextEdit* list_text_constant = new QTextEdit;
	connect(list_text_constant, &QTextEdit::textChanged,
	        std::bind(store_list_f, list_text_constant));
	QTextEdit* list_text_singlestate = new QTextEdit;
	connect(list_text_singlestate, &QTextEdit::textChanged,
	        std::bind(store_list_f, list_text_singlestate));
	QTextEdit* states_list_text = new QTextEdit;
	list_stacked_text->addWidget(list_text_constant);
	list_stacked_text->addWidget(list_text_singlestate);
	list_stacked_text->addWidget(states_list_text);

	list_layout->addWidget(list_label);
	list_layout->addWidget(list_stacked_text);
	QWidget* list_widget = new QWidget;
	list_widget->setLayout(list_layout);
	list_widget->setVisible(false);

	splitter->addWidget(tree);
	splitter->addWidget(list_widget);
	splitter->setStretchFactor(0, 2);
	splitter->setStretchFactor(1, 1);

	connect(tree, &PVRush::PVERFTreeView::current_changed,
	        [=](const QModelIndex& current, const QModelIndex& /*previous*/) {
		        size_t index = 0;
		        static const std::unordered_map<std::string, size_t> text_index_map = {
		            {"post.constant.entityresults.NODE", index++},
		            {"post.singlestate.entityresults.NODE", index++},
		            {"post.singlestate.states", index++}};
		        if (current.isValid()) {
			        PVERFTreeItem* item = static_cast<PVERFTreeItem*>(current.internalPointer());
					const std::unordered_map<PVERFTreeItem::EType, QString> list_label_map {
						std::make_pair(PVERFTreeItem::EType::NODE, "Nodes"),
						std::make_pair(PVERFTreeItem::EType::STATES, "States")
					};
			        if (item->type() == PVERFTreeItem::EType::NODE or
			            item->type() == PVERFTreeItem::EType::STATES) {
				        QWidget* widget = list_stacked_text->widget(
				            text_index_map.at(item->path().toStdString()));
				        widget->setProperty("index", current);
				        list_stacked_text->setCurrentWidget(widget);
				        list_widget->setVisible(true);
						list_label->setText(list_label_map.at(item->type()) + " list:");
				        return;
			        }
		        }
		        list_widget->setVisible(false);
	        });

	connect(tree, &PVRush::PVERFTreeView::model_changed,
	        [&]() { _status_bar_needs_refresh = true; });

	QDialogButtonBox* dialog_buttons =
	    new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	connect(dialog_buttons, &QDialogButtonBox::accepted, [this]() {
		size_t bytes_count = _erf->memory_size(_model->save(), _paths.size());
		if (bytes_count > PVCore::available_memory()) {
			QMessageBox::StandardButton ret = QMessageBox::warning(
			    this, "Not enough available memory", "Continuing could lead to a crash.",
			    QMessageBox::Ok | QMessageBox::Cancel);
			if (ret == QMessageBox::Cancel) {
				return;
			}
		}
		// Check if states are selected
		const rapidjson::Document& json = _model->save();
		if (rapidjson::Pointer("/post/singlestate/entityresults").Get(json)
		    and not rapidjson::Pointer("/post/singlestate/states").Get(json)) {
			QMessageBox::critical(
			this, "No states selected", "Please, select one or more states to continue",
			QMessageBox::Ok);
			return;
		}
		accept();
	});
	connect(dialog_buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

	QVBoxLayout* vlayout = new QVBoxLayout;

	vlayout->addWidget(splitter);
	vlayout->addWidget(dialog_buttons);
	vlayout->addWidget(status_bar);

	// update status bar every 1 sec
	QTimer* status_bar_timer = new QTimer(this);
	connect(status_bar_timer, &QTimer::timeout, [this, status_bar]() {
		if (_status_bar_needs_refresh) {
			size_t bytes_count = _erf->memory_size(_model->save(), _paths.size());
			QString bytes_count_str = PVWidgets::PVUtils::bytes_to_human_readable(bytes_count);
			status_bar->showMessage(QString("Approximative amount of RAM : ") + bytes_count_str);
			_status_bar_needs_refresh = false;
		}
	});
	status_bar_timer->start(1000);

	setLayout(vlayout);
}

std::vector<std::tuple<rapidjson::Document, std::string, PVRush::PVFormat>>
PVRush::PVERFParamsWidget::get_sources_info() const
{
	return _erf->get_sources_info(_model->save(), _paths.size() > 1);
}