#include <PVNrawListingWidget.h>
#include <PVNrawListingModel.h>

#include <QLabel>
#include <QTableView>
#include <QVBoxLayout>

#include <pvcore/general.h>

PVInspector::PVNrawListingWidget::PVNrawListingWidget(PVNrawListingModel* nraw_model, QWidget* parent) :
	QWidget(parent),
	_nraw_model(nraw_model)
{
	QVBoxLayout* main_layout = new QVBoxLayout();

	// NRAW table view
	QTableView* nraw_table = new QTableView();
	nraw_table->setModel(_nraw_model);

	// "Mini-extractor" for this NRAW
	QHBoxLayout* ext_layout = new QHBoxLayout();
	ext_layout->addWidget(new QLabel("Preview from line "));

	_ext_start = new QLineEdit();
	QIntValidator *iv_start = new QIntValidator();
	iv_start->setBottom(0);
	iv_start->setTop(PICVIZ_LINES_MAX);
	_ext_start->setValidator(iv_start);
	_ext_start->setText(QString::number(FORMATBUILDER_EXTRACT_START_DEFAULT));
	ext_layout->addWidget(_ext_start);

	ext_layout->addWidget(new QLabel(" to line "));

	_ext_end = new QLineEdit();
	QIntValidator *iv_end = new QIntValidator();
	iv_end->setBottom(100);
	iv_end->setTop(PICVIZ_LINES_MAX);
	_ext_end->setValidator(iv_start);
	_ext_end->setText(QString::number(FORMATBUILDER_EXTRACT_END_DEFAULT));
	ext_layout->addWidget(_ext_end);

	_btn_preview = new QPushButton("Preview");
	ext_layout->addWidget(_btn_preview);

	main_layout->addWidget(nraw_table);
	main_layout->addItem(ext_layout);

	setLayout(main_layout);
}

void PVInspector::PVNrawListingWidget::connect_preview(QObject* receiver, const char* slot)
{
	connect(_btn_preview, SIGNAL(clicked()), receiver, slot);
}

void PVInspector::PVNrawListingWidget::get_ext_args(PVRow& start, PVRow& end)
{
	start = _ext_start->text().toULongLong();
	end = _ext_end->text().toULongLong();
	if (end < 10) {
		end = 10;
	}
	if (end <= start) {
		start = 0;
	}
}
