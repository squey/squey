/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvbase/general.h>

#include <PVNrawListingWidget.h>
#include <PVNrawListingModel.h>

#include <QLabel>
#include <QVBoxLayout>
#include <QPoint>
#include <QMenu>
#include <QAction>

PVInspector::PVNrawListingWidget::PVNrawListingWidget(PVNrawListingModel* nraw_model,
                                                      QWidget* parent)
    : QWidget(parent), _nraw_model(nraw_model)
{
	auto main_layout = new QVBoxLayout();

	// Current source display
	auto src_layout = new QHBoxLayout();
	_src_label = new QLabel();
	src_layout->addWidget(_src_label);

	// NRAW table view
	_nraw_table = new QTableView();
	_nraw_table->setModel(_nraw_model);

	// Context menu for the NRAW table
	_ctxt_menu = new QMenu(this);
	QAction* act_set_axis_name = new QAction(tr("Set axes' name based on this row"), _ctxt_menu);
	connect(act_set_axis_name, SIGNAL(triggered()), this, SLOT(set_axes_name_selected_row_Slot()));
	_ctxt_menu->addAction(act_set_axis_name);
	QAction* act_detect_type =
	    new QAction(tr("Automatically detect axes' type based on this row"), _ctxt_menu);
	connect(act_detect_type, SIGNAL(triggered()), this, SLOT(set_axes_type_selected_row_Slot()));
	//_ctxt_menu->addAction(act_detect_type);

	connect(_nraw_table, SIGNAL(customContextMenuRequested(const QPoint&)), this,
	        SLOT(nraw_custom_menu_Slot(const QPoint&)));
	_nraw_table->setContextMenuPolicy(Qt::CustomContextMenu);

	// "Mini-extractor" for this NRAW
	auto ext_layout = new QHBoxLayout();
	ext_layout->addWidget(new QLabel("Preview from line "));

	_ext_start = new QSpinBox(this);
	_ext_start->setRange(0, std::numeric_limits<int32_t>::max());
	_ext_start->setValue(FORMATBUILDER_EXTRACT_START_DEFAULT);
	ext_layout->addWidget(_ext_start);

	ext_layout->addWidget(new QLabel(" to line "));

	_ext_end = new QSpinBox(this);
	_ext_end->setRange(0, std::numeric_limits<int32_t>::max());
	_ext_end->setValue(FORMATBUILDER_EXTRACT_END_DEFAULT);
	ext_layout->addWidget(_ext_end);

	// Ensure begin and end are consistent
	connect(_ext_start, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),
	        [this](int v) {
		        if (v > _ext_end->value()) {
			        _ext_end->setValue(v);
		        }
		    });
	connect(_ext_end, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), [this](int v) {
		if (v < _ext_start->value()) {
			_ext_start->setValue(v);
		}
	});

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

void PVInspector::PVNrawListingWidget::connect_axes_name(QObject* receiver, const char* slot)
{
	connect(this, SIGNAL(set_axes_name_from_nraw(int)), receiver, slot);
}

void PVInspector::PVNrawListingWidget::connect_axes_type(QObject* receiver, const char* slot)
{
	connect(this, SIGNAL(set_axes_type_from_nraw(int)), receiver, slot);
}

void PVInspector::PVNrawListingWidget::get_ext_args(PVRow& start, PVRow& end)
{
	start = _ext_start->value();
	end = _ext_end->value();
}

void PVInspector::PVNrawListingWidget::set_last_input(PVRush::PVInputType_p in_t,
                                                      PVRush::PVInputDescription_p input)
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
	for (int col = 1; col < _nraw_table->model()->columnCount(); col++) {
		_nraw_table->resizeColumnToContents(col);
	}
}

void PVInspector::PVNrawListingWidget::select_column(PVCol col)
{
	PVLOG_DEBUG("(PVNrawListingWidget) select column %d\n", col);
	_nraw_model->set_selected_column(col);
	_nraw_model->sel_visible(true);

	// Scroll to that column, but keep the current row
	QModelIndex first_visible_idx = _nraw_table->indexAt(QPoint(0, 0));
	QModelIndex col_idx = _nraw_model->index(first_visible_idx.row(), col);
	_nraw_table->scrollTo(col_idx, QAbstractItemView::PositionAtTop);
}

void PVInspector::PVNrawListingWidget::unselect_column()
{
	PVLOG_DEBUG("(PVNrawListingWidget) select no column\n");
	_nraw_model->sel_visible(false);
}

void PVInspector::PVNrawListingWidget::set_axes_name_selected_row_Slot()
{
	int row = get_selected_row();
	Q_EMIT set_axes_name_from_nraw(row);
}

void PVInspector::PVNrawListingWidget::set_axes_type_selected_row_Slot()
{
	int row = get_selected_row();
	Q_EMIT set_axes_type_from_nraw(row);
}

int PVInspector::PVNrawListingWidget::get_selected_row()
{
	return _nraw_table->currentIndex().row();
}

void PVInspector::PVNrawListingWidget::nraw_custom_menu_Slot(const QPoint&)
{
	if (_nraw_table->model()->rowCount() == 0) {
		return;
	}

	_ctxt_menu->exec(QCursor::pos());
}

void PVInspector::PVNrawListingWidget::mark_row_as_invalid(size_t row_index)
{
	_nraw_table->setSpan(row_index, 0, 1, _nraw_table->model()->columnCount());
}
