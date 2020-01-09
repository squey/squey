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

#include <QTreeView>
#include <QHBoxLayout>
#include <QSplitter>
#include <QTextEdit>
#include <QPushButton>
#include <QMessageBox>

#include <pvlogger.h>

///
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
///

PVRush::PVERFParamsWidget::PVERFParamsWidget(PVInputTypeERF const* in_t, QWidget* parent)
{
	QHBoxLayout* layout = new QHBoxLayout();

	QSplitter* splitter = new QSplitter(Qt::Horizontal);

	static constexpr const char path[] = "/srv/logs/VW/BOOST_fill_sol_V01_OPT01_r02g_VV1.hdf5";

	_model.reset(new PVRush::PVERFTreeModel(path));
	PVRush::PVERFTreeView* tree = new PVRush::PVERFTreeView(_model.get(), parent);
	tree->setSelectionMode(QAbstractItemView::MultiSelection);
	tree->setAlternatingRowColors(true);
	tree->expandAll();

	QTextEdit* text = new QTextEdit;
	QPushButton* export_btn = new QPushButton(">");
	QPushButton* import_btn = new QPushButton("<");

	QWidget* singlestate_widget = new QWidget;

	QHBoxLayout* hlayout = new QHBoxLayout;
	hlayout->addWidget(text);
	hlayout->addWidget(export_btn);
	hlayout->addWidget(import_btn);

	singlestate_widget->setLayout(hlayout);

	connect(export_btn, &QPushButton::clicked, [=]() {
		rapidjson::Document doc = _model->save();

		rapidjson::StringBuffer buffer;
		buffer.Clear();
		rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
		doc.Accept(writer);
		text->setText(buffer.GetString());
	});

	connect(import_btn, &QPushButton::clicked, [=]() {
		const QString& json = text->toPlainText();
		rapidjson::Document selection;
		selection.Parse<0>(json.toStdString().c_str());
		tree->select(selection);
	});

	splitter->addWidget(tree);
	splitter->addWidget(singlestate_widget);

	layout->addWidget(splitter);

	setLayout(layout);
}

std::vector<QDomDocument> PVRush::PVERFParamsWidget::get_formats()
{
	return PVERFAPI(_model->path().toStdString()).get_formats_from_selected_nodes(_model->save());
}

QString PVRush::PVERFParamsWidget::path() const
{
	return _model->path();
}

rapidjson::Document PVRush::PVERFParamsWidget::get_selected_nodes() const
{
	return _model->save();
}