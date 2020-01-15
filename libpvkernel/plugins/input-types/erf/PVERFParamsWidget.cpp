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

#include <QTreeView>
#include <QHBoxLayout>
#include <QSplitter>
#include <QTextEdit>
#include <QPushButton>
#include <QMessageBox>
#include <QGuiApplication>
#include <QScreen>
#include <QDialogButtonBox>

#include <pvlogger.h>

///
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
///

PVRush::PVERFParamsWidget::PVERFParamsWidget(PVInputTypeERF const* in_t, QWidget* parent)
{
	QScreen* screen = QGuiApplication::primaryScreen();
	QRect screen_geometry = screen->geometry();
	int height = screen_geometry.height();
	int width = screen_geometry.width();

	QHBoxLayout* layout = new QHBoxLayout();

	QSplitter* splitter = new QSplitter(Qt::Horizontal);
	splitter->setFixedSize(QSize(width / 3, height / 2));

	// static constexpr const char path[] = "/srv/logs/VW/BOOST_fill_sol_V01_OPT01_r02g_VV1.hdf5";

	PVWidgets::PVFileDialog fdialog(this);
	fdialog.setNameFilter("ERF files (*.erf, *.erfh5");
	fdialog.setWindowTitle("Open ERF file");
	QString erf_path;
	if (fdialog.exec() == QDialog::Accepted) {
		erf_path = fdialog.selectedFiles().at(0);
	}

	_model.reset(new PVRush::PVERFTreeModel(erf_path));
	PVRush::PVERFTreeView* tree = new PVRush::PVERFTreeView(_model.get(), parent);
	tree->setSelectionMode(QAbstractItemView::MultiSelection);
	tree->setAlternatingRowColors(true);
	tree->expandAll();

#if 0
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
#endif

	splitter->addWidget(tree);
	//splitter->addWidget(singlestate_widget);

	QDialogButtonBox* dialog_buttons =
	    new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	connect(dialog_buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(dialog_buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

	QVBoxLayout* vlayout = new QVBoxLayout;

	vlayout->addWidget(splitter);
	vlayout->addWidget(dialog_buttons);

	setLayout(vlayout);
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