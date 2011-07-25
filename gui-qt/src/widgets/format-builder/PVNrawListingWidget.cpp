#include <PVNrawListingWidget.h>
#include <PVNrawListingModel.h>

#include <QLabel>
#include <QVBoxLayout>

#include <pvcore/general.h>

PVInspector::PVNrawListingWidget::PVNrawListingWidget(PVNrawListingModel* nraw_model, QWidget* parent) :
	QWidget(parent),
	_nraw_model(nraw_model)
{
	QVBoxLayout* main_layout = new QVBoxLayout();

	// Current source display
	QHBoxLayout* src_layout = new QHBoxLayout();
	_src_label = new QLabel();
	src_layout->addWidget(_src_label);

	// NRAW table view
	_nraw_table = new QTableView();
	_nraw_table->setModel(_nraw_model);

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
	_btn_preview->setAutoDefault(false);

	main_layout->addItem(src_layout);	
	main_layout->addWidget(_nraw_table);
	main_layout->addItem(ext_layout);

	set_last_input();

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

void PVInspector::PVNrawListingWidget::set_last_input(PVRush::PVInputType_p in_t, PVCore::PVArgument input)
{
	if (!in_t) {
		_src_label->hide();
		_btn_preview->setEnabled(false);
		return;
	}
	QString txt = tr("This is a preview of the normalisation process for the input ");
	txt += in_t->human_name_of_input(input);
	_src_label->setText(txt);
	_src_label->show();
	_btn_preview->setEnabled(true);
}

void PVInspector::PVNrawListingWidget::resize_columns_content()
{
	_nraw_table->resizeColumnsToContents();
}

void PVInspector::PVNrawListingWidget::select_column(PVCol col)
{
	PVLOG_DEBUG("(PVNrawListingWidget) select column %d\n", col);
	_nraw_model->set_selected_column(col);
	_nraw_model->sel_visible(true);
}

void PVInspector::PVNrawListingWidget::unselect_column()
{
	PVLOG_DEBUG("(PVNrawListingWidget) select no column\n");
	_nraw_model->sel_visible(false);
}
