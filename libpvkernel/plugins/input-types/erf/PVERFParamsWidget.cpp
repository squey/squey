/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2019
 */

#include "PVERFParamsWidget.h"
#include "PVERFTreeView.h"
#include "../../common/erf/PVERFAPI.h"

#include <pvkernel/core/serialize_numbers.h>
#include <pvkernel/widgets/PVFileDialog.h>

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

#include <pvlogger.h>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

PVRush::PVERFParamsWidget::PVERFParamsWidget(PVInputTypeERF const* in_t, QWidget* parent)
{
	QScreen* screen = QGuiApplication::primaryScreen();
	QRect screen_geometry = screen->geometry();
	int height = screen_geometry.height();
	int width = screen_geometry.width();

	QHBoxLayout* layout = new QHBoxLayout();

	QSplitter* splitter = new QSplitter(Qt::Horizontal);
	splitter->setFixedSize(QSize(width / 3, height / 2));

	QString erf_path = PVWidgets::PVFileDialog::getOpenFileName(this, tr("Open ERF file"), "",
	                                                            tr("ERF files (*.erf, *.erfh5"));

	if (erf_path.isEmpty()) {
		setResult(QDialog::Rejected);
		return;
	}
	setResult(QDialog::Accepted);

	_model.reset(new PVRush::PVERFTreeModel(erf_path));
	PVRush::PVERFTreeView* tree = new PVRush::PVERFTreeView(_model.get(), parent);
	tree->setSelectionMode(QAbstractItemView::MultiSelection);
	tree->setAlternatingRowColors(true);
	tree->expandAll();

	QVBoxLayout* nodes_list_layout = new QVBoxLayout;
	QLabel* nodes_list_label = new QLabel("Nodes list:");

	auto store_list_f = [](QTextEdit* text_edit) {
		// sender() in lambda function is always returning nullptr
		QModelIndex index = text_edit->property("index").toModelIndex();
		PVERFTreeItem* item = static_cast<PVERFTreeItem*>(index.internalPointer());
		item->set_user_data(text_edit->toPlainText());
	};

	QStackedWidget* nodes_list_stacked_text = new QStackedWidget;
	QTextEdit* nodes_list_text_constant = new QTextEdit;
	connect(nodes_list_text_constant, &QTextEdit::textChanged,
	        std::bind(store_list_f, nodes_list_text_constant));
	QTextEdit* nodes_list_text_singlestate = new QTextEdit;
	connect(nodes_list_text_singlestate, &QTextEdit::textChanged,
	        std::bind(store_list_f, nodes_list_text_singlestate));
	QTextEdit* states_list_text = new QTextEdit;
	nodes_list_stacked_text->addWidget(nodes_list_text_constant);
	nodes_list_stacked_text->addWidget(nodes_list_text_singlestate);
	nodes_list_stacked_text->addWidget(states_list_text);

	nodes_list_layout->addWidget(nodes_list_label);
	nodes_list_layout->addWidget(nodes_list_stacked_text);
	QWidget* nodes_list_widget = new QWidget;
	nodes_list_widget->setLayout(nodes_list_layout);
	nodes_list_widget->setVisible(false);

	splitter->addWidget(tree);
	splitter->addWidget(nodes_list_widget);
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
			        if (item->type() == PVERFTreeItem::EType::NODE or
			            item->type() == PVERFTreeItem::EType::STATES) {
				        QWidget* widget = nodes_list_stacked_text->widget(
				            text_index_map.at(item->path().toStdString()));
				        widget->setProperty("index", current);
				        nodes_list_stacked_text->setCurrentWidget(widget);
				        nodes_list_widget->setVisible(true);
				        return;
			        }
		        }
		        nodes_list_widget->setVisible(false);
	        });

	QDialogButtonBox* dialog_buttons =
	    new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	connect(dialog_buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(dialog_buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

	QVBoxLayout* vlayout = new QVBoxLayout;

	vlayout->addWidget(splitter);
	vlayout->addWidget(dialog_buttons);

	setLayout(vlayout);
}

std::vector<std::tuple<rapidjson::Document, std::string, PVRush::PVFormat>>
PVRush::PVERFParamsWidget::get_sources_info() const
{
	return PVERFAPI(_model->path().toStdString()).get_sources_info(_model->save());
}

QString PVRush::PVERFParamsWidget::path() const
{
	return _model->path();
}