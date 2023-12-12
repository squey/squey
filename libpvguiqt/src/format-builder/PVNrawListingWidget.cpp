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

#include <pvbase/general.h>

#include <PVNrawListingWidget.h>
#include <PVNrawListingModel.h>

#include <QLabel>
#include <QVBoxLayout>
#include <QPoint>
#include <QMenu>
#include <QAction>
#include <QHeaderView>

#include <pvlogger.h>

App::PVNrawListingWidget::PVNrawListingWidget(PVNrawListingModel* nraw_model,
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
	_nraw_table->setSelectionBehavior(QAbstractItemView::SelectRows);
	_nraw_table->setSelectionMode(QAbstractItemView::SingleSelection);

	// Context menu for the NRAW table
	_ctxt_menu = new QMenu(this);
	auto* act_set_axis_name = new QAction(tr("Set axes' name based on this row"), _ctxt_menu);
	connect(act_set_axis_name, &QAction::triggered, this,
	        &PVNrawListingWidget::set_axes_name_selected_row_Slot);
	_ctxt_menu->addAction(act_set_axis_name);
	auto* act_detect_type =
	    new QAction(tr("Automatically detect axes' type based on this row"), _ctxt_menu);
	connect(act_detect_type, &QAction::triggered, this,
	        &PVNrawListingWidget::set_axes_type_selected_row_Slot);
	//_ctxt_menu->addAction(act_detect_type);

	connect(_nraw_table, &QTableView::customContextMenuRequested, this,
	        &PVNrawListingWidget::nraw_custom_menu_Slot);
	_nraw_table->setContextMenuPolicy(Qt::CustomContextMenu);

	// "Mini-extractor" for this NRAW
	auto ext_layout = new QHBoxLayout();
	ext_layout->addWidget(new QLabel("Preview"));

	_ext_count = new PVGuiQt::PVLocalizedSpinBox(this);
	_ext_count->setRange(0, std::numeric_limits<int32_t>::max());
	_ext_count->setValue(FORMATBUILDER_EXTRACT_END_DEFAULT);
	ext_layout->addWidget(_ext_count);

	ext_layout->addWidget(new QLabel(" row(s), starting from row #"));

	_ext_start = new PVGuiQt::PVLocalizedSpinBox(this);
	_ext_start->setRange(1, std::numeric_limits<int32_t>::max());
	_ext_start->setValue(FORMATBUILDER_EXTRACT_START_DEFAULT);
	ext_layout->addWidget(_ext_start);

	_btn_preview = new QPushButton("Preview");
	ext_layout->addWidget(_btn_preview);
	_btn_preview->setAutoDefault(false);

	auto autodetect_layout = new QHBoxLayout();
	autodetect_layout->addWidget(new QLabel("autodetect "));

	_autodetect_count = new PVGuiQt::PVLocalizedSpinBox(this);
	_autodetect_count->setRange(0, std::numeric_limits<int32_t>::max());
	_autodetect_count->setValue(FORMATBUILDER_EXTRACT_END_DEFAULT);
	autodetect_layout->addWidget(_autodetect_count);

	autodetect_layout->addWidget(new QLabel(" row(s), starting from row # "));

	_autodetect_start = new PVGuiQt::PVLocalizedSpinBox(this);
	_autodetect_start->setRange(1, std::numeric_limits<int32_t>::max());
	_autodetect_start->setValue(FORMATBUILDER_EXTRACT_START_DEFAULT);
	autodetect_layout->addWidget(_autodetect_start);

	_btn_autodetect = new QPushButton("autodetect axes types");
	autodetect_layout->addWidget(_btn_autodetect);
	_btn_autodetect->setAutoDefault(false);

	main_layout->addItem(src_layout);
	main_layout->addWidget(_nraw_table);
	main_layout->addItem(autodetect_layout);
	main_layout->addItem(ext_layout);
	set_last_input();

	setLayout(main_layout);
}

void App::PVNrawListingWidget::get_ext_args(PVRow& start, PVRow& end)
{
	start = _ext_start->value() - 1;
	end = start + _ext_count->value() - 1;
}

void App::PVNrawListingWidget::get_autodetect_args(PVRow& start, PVRow& end)
{
	start = _autodetect_start->value() - 1;
	end = start + _autodetect_count->value() - 1;
}

void App::PVNrawListingWidget::set_autodetect_count(PVRow count)
{
	_autodetect_count->setValue(count);
}

void App::PVNrawListingWidget::set_last_input(PVRush::PVInputType_p in_t,
                                                      PVRush::PVInputDescription_p input)
{
	if (!in_t) {
		_src_label->hide();
		_btn_preview->setEnabled(false);
		_btn_autodetect->setEnabled(false);
		return;
	}
	QString txt = tr("This is a preview of the normalisation process for the input ");
	txt += in_t->human_name_of_input(input);
	_src_label->setText(txt);
	_src_label->show();
	_btn_preview->setEnabled(true);
	_btn_autodetect->setEnabled(true);
}

void App::PVNrawListingWidget::resize_columns_content()
{
	for (int col = 1; col < _nraw_table->model()->columnCount(); col++) {
		_nraw_table->resizeColumnToContents(col);
	}
}

void App::PVNrawListingWidget::select_header(PVCol column)
{
	_nraw_model->set_selected_column(column);
	_nraw_model->sel_visible(true);
}

void App::PVNrawListingWidget::set_error_message(QString error)
{
	_nraw_table->setVisible(false);
	_src_label->setStyleSheet("QLabel { color : red; }");
	_src_label->setText(error);
}

void App::PVNrawListingWidget::unset_error_message()
{
	_src_label->setStyleSheet("");
	_nraw_table->setVisible(true);
}

void App::PVNrawListingWidget::select_column(PVCol col)
{
	PVLOG_DEBUG("(PVNrawListingWidget) select column %d\n", col);
	_nraw_model->set_selected_column(col);
	_nraw_model->sel_visible(true);

	// Scroll to that column, but keep the current row
	QModelIndex first_visible_idx = _nraw_table->indexAt(QPoint(0, 0));
	QModelIndex col_idx = _nraw_model->index(first_visible_idx.row(), col);
	_nraw_table->scrollTo(col_idx, QAbstractItemView::PositionAtTop);
}

void App::PVNrawListingWidget::unselect_column()
{
	PVLOG_DEBUG("(PVNrawListingWidget) select no column\n");
	_nraw_model->sel_visible(false);
}

void App::PVNrawListingWidget::set_axes_name_selected_row_Slot()
{
	int row = get_selected_row();
	Q_EMIT set_axes_name_from_nraw(row);
}

void App::PVNrawListingWidget::set_axes_type_selected_row_Slot()
{
	int row = get_selected_row();
	Q_EMIT set_axes_type_from_nraw(row);
}

int App::PVNrawListingWidget::get_selected_row()
{
	return _nraw_table->currentIndex().row();
}

void App::PVNrawListingWidget::nraw_custom_menu_Slot(const QPoint&)
{
	if (_nraw_table->model()->rowCount() == 0) {
		return;
	}

	_ctxt_menu->exec(QCursor::pos());
}

void App::PVNrawListingWidget::mark_row_as_invalid(size_t row_index)
{
	_nraw_table->setSpan(row_index, 0, 1, _nraw_table->model()->columnCount());
}
