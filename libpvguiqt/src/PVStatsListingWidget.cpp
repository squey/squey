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

#include <pvguiqt/PVStatsListingWidget.h>
#include <pvguiqt/PVDisplayViewDistinctValues.h>
#include <pvkernel/widgets/PVModdedIcon.h>

#include <squey/PVSource.h>

#include <pvkernel/core/qobject_helpers.h>

#include <pvcop/db/algo.h>

#include <QApplication>
#include <QCursor>
#include <QDialog>
#include <QMenu>
#include <QMimeData>
#include <QPushButton>
#include <QPushButton>
#include <QScrollBar>
#include <QVBoxLayout>

constexpr int QTABLEWIDGET_OFFSET = 4;

// Originally from
// http://stackoverflow.com/questions/8766633/how-to-determine-the-correct-size-of-a-qtablewidget
static uint32_t compute_qtablewidget_height(QTableWidget* stats)
{
	int h = stats->horizontalHeader()->height() + QTABLEWIDGET_OFFSET;
	for (int i = 0; i < stats->verticalHeader()->count(); i++) {
		if (!stats->isRowHidden(i)) {
			h += stats->rowHeight(i);
		}
	}

	return h;
}

/******************************************************************************
 *
 * PVGuiQt::PVStatsListingWidget
 *
 *****************************************************************************/

PVGuiQt::PVStatsListingWidget::PVStatsListingWidget(PVGuiQt::PVListingView* listing_view)
    : _listing_view(listing_view)
	, _help_widget(this)
{
	_params.clear();

	auto* main_layout = new QVBoxLayout();

	_stats_panel = new QTableWidget(this);
	_stats_panel->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	_stats_panel->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	_stats_panel->viewport()->setFocusPolicy(Qt::NoFocus);
	//_stats_panel->hide();
	QStringList horizontal_header_labels;
	for (PVCol col(0); col < _listing_view->horizontalHeader()->count(); col++) {
		horizontal_header_labels
		    << _listing_view->model()->headerData(col, Qt::Horizontal).toString();
	}
	_stats_panel->setHorizontalHeaderLabels(horizontal_header_labels);
	_stats_panel->horizontalHeader()->setStretchLastSection(true);
	_stats_panel->horizontalHeader()->hide();
	_stats_panel->setSelectionMode(QTableWidget::NoSelection);

	main_layout->setSpacing(0);
	main_layout->setContentsMargins(0, 0, 0, 0);

	auto* hide_button = new QPushButton("...");
	hide_button->setToolTip(tr("Toggle stats panel visibility"));
	hide_button->setMaximumHeight(10);
	hide_button->setFlat(true);
	connect(hide_button, &QAbstractButton::clicked, this,
	        &PVStatsListingWidget::toggle_stats_panel_visibility);

	main_layout->addWidget(_listing_view);
	main_layout->addWidget(hide_button);
	main_layout->addWidget(_stats_panel);

	setLayout(main_layout);

	connect(_listing_view->horizontalHeader(), &QHeaderView::sectionResized, this,
	        &PVStatsListingWidget::update_header_width);
	_stats_panel->setVerticalHeader(new __impl::PVVerticalHeaderView(this));
	_stats_panel->verticalHeader()->viewport()->installEventFilter(this);
	connect(_listing_view, &PVListingView::resized, this, &PVStatsListingWidget::resize_panel);
	connect(_listing_view->horizontalScrollBar(), &QAbstractSlider::valueChanged, this,
	        &PVStatsListingWidget::update_scrollbar_position);

	// Observe selection to handle automatic refresh mode
	Squey::PVView& view_sp = _listing_view->lib_view();
	view_sp._update_output_selection.connect(
	    sigc::mem_fun(*this, &PVGuiQt::PVStatsListingWidget::selection_changed));

	// Observe layerstack to handle automatic refresh mode
	view_sp._update_output_layer.connect(
	    sigc::mem_fun(*this, &PVGuiQt::PVStatsListingWidget::selection_changed));

	// Observer axes combination changes
	view_sp._axis_combination_updated.connect(
	    sigc::mem_fun(*this, &PVGuiQt::PVStatsListingWidget::axes_comb_changed));

	// Define help
	_help_widget.hide();

	_help_widget.initTextFromFile("listing view's help", ":help-style");
	_help_widget.addTextFromFile(":help-selection");
	_help_widget.addTextFromFile(":help-layers");
	_help_widget.newColumn();
	_help_widget.addTextFromFile(":help-lines");
	_help_widget.addTextFromFile(":help-application");

	_help_widget.newTable();
	_help_widget.addTextFromFile(":help-mouse-listing-view");
	_help_widget.newColumn();
	_help_widget.addTextFromFile(":help-shortcuts-listing-view");
	_help_widget.finalizeText();

	init_plugins();
	create_vhead_ctxt_menu();

	resize_panel();
	refresh();
	
}

void PVGuiQt::PVStatsListingWidget::create_vhead_ctxt_menu()
{
	_vhead_ctxt_menu = new QMenu(this);

	for (int row = 0; row < _stats_panel->rowCount(); row++) {
		const QString& section_text = _stats_panel->verticalHeaderItem(row)->toolTip();
		auto* act = new QAction(section_text, this);
		act->setCheckable(true);
		act->setEnabled(_stats_panel->isRowHidden(row));
		act->setChecked(!_stats_panel->isRowHidden(row));
		act->setData(row);
		connect(act, &QAction::triggered, this, &PVStatsListingWidget::plugin_visibility_toggled);
		_vhead_ctxt_menu->addAction(act);
	}
}

void PVGuiQt::PVStatsListingWidget::plugin_visibility_toggled(bool checked)
{
	auto* act = (QAction*)sender();
	assert(act);
	int row = act->data().toInt();
	assert(row < _stats_panel->rowCount());
	if (checked) {
		_stats_panel->showRow(row);
	} else {
		_stats_panel->hideRow(row);
	}
	_stats_panel->setMaximumHeight(compute_qtablewidget_height(_stats_panel));
};

void PVGuiQt::PVStatsListingWidget::resize_listing_column_if_needed(int col)
{
	int cell_max_size = 0;
	for (int row = 0; row < _stats_panel->rowCount(); row++) {
		__impl::PVCellWidgetBase* cell_widget =
		    ((__impl::PVCellWidgetBase*)_stats_panel->cellWidget(row, col));
		assert(cell_widget);
		cell_max_size = std::max(cell_max_size, cell_widget->minimum_size());
	}

	int listing_col_width = _listing_view->columnWidth(col);
	if (listing_col_width < cell_max_size) {
		_listing_view->setColumnWidth(col, cell_max_size);
	}
}

void PVGuiQt::PVStatsListingWidget::init_plugins()
{
	for (PVCombCol col(0); col < _listing_view->horizontalHeader()->count(); col++) {
		_stats_panel->insertColumn(col);
	}

	_row_distinct =
	    init_plugin<__impl::PVUniqueValuesCellWidget>("Distinct values", PVModdedIcon("brackets-curly"), /* visible = */ true);
	_row_sum = init_plugin<__impl::PVSumCellWidget>("Sum", PVModdedIcon("sigma"), /* visible = */ false);
	_row_min = init_plugin<__impl::PVMinCellWidget>("Min", PVModdedIcon("arrow-down-to-line"), /* visible = */ false);
	_row_max = init_plugin<__impl::PVMaxCellWidget>("Max", PVModdedIcon("arrow-up-to-line"), /* visible = */ false);
	_row_avg = init_plugin<__impl::PVAverageCellWidget>("Average", PVModdedIcon("average-by"), /* visible = */ false);

	for (PVCombCol col(0); col < _listing_view->horizontalHeader()->count(); col++) {
		_stats_panel->setColumnWidth(col, _listing_view->horizontalHeader()->sectionSize(col));
	}
}

void PVGuiQt::PVStatsListingWidget::refresh()
{
	sync_vertical_headers();

	for (PVCombCol col(0); col < _stats_panel->columnCount(); col++) {
		for (int row = 0; row < _stats_panel->rowCount(); row++) {
			((__impl::PVCellWidgetBase*)_stats_panel->cellWidget(row, col))->auto_refresh();
		}
	}
}

void PVGuiQt::PVStatsListingWidget::sync_vertical_headers()
{
	// Sync tables vertical header width
	int stats_header_width = _stats_panel->verticalHeader()->width();
	int listing_header_width = _listing_view->verticalHeader()->width();

	if (stats_header_width > listing_header_width) {
		_listing_view->verticalHeader()->setFixedWidth(stats_header_width);
		QMetaObject::invokeMethod(_listing_view, "updateGeometries");
	} else {
		_stats_panel->verticalHeader()->setFixedWidth(listing_header_width);
		QMetaObject::invokeMethod(_stats_panel, "updateGeometries");
	}
}

void PVGuiQt::PVStatsListingWidget::toggle_stats_panel_visibility()
{
	_stats_panel->setVisible(!_stats_panel->isVisible());
}

void PVGuiQt::PVStatsListingWidget::update_header_width(int column,
                                                        int /*old_width*/,
                                                        int new_width)
{
	_stats_panel->setColumnWidth(column, new_width);
}

void PVGuiQt::PVStatsListingWidget::resize_panel()
{
	_stats_panel->setMaximumHeight(compute_qtablewidget_height(_stats_panel));
	for (PVCombCol col(0); col < _stats_panel->columnCount(); col++) {
		_stats_panel->setColumnWidth(col, _listing_view->columnWidth(col));
	}
}

void PVGuiQt::PVStatsListingWidget::update_scrollbar_position()
{
	// Difference between QScrollBar::value() and QScrollBar::sliderPosition():
	// From Qt documentation: If tracking is enabled (the default), the slider
	// emits the valueChanged() signal while the slider is being dragged.
	//                        If tracking is disabled, the slider emits the
	//                        valueChanged() signal only when the user releases
	//                        the slider.

	// In order to avoid an offset between the stats and listing tables when the
	// vertical scrollbar is on the rightmost position
	// the maximum width of the stats panel is in this case reduced of by the
	// vertical scrollbar width. (fix bug #238)
	if (_listing_view->horizontalScrollBar()->sliderPosition() ==
	    _listing_view->horizontalScrollBar()->maximum()) {
		_old_maximum_width = _stats_panel->maximumSize().width();
		_stats_panel->setMaximumWidth(_stats_panel->minimumSize().width() +
		                              _listing_view->horizontalScrollBar()->width() +
		                              QTABLEWIDGET_OFFSET / 2);
		_maxed = true;
	} else {
		if (_maxed) {
			_stats_panel->setMaximumWidth(_old_maximum_width);
			_maxed = false;
		}
	}
	_stats_panel->horizontalScrollBar()->setSliderPosition(
	    _listing_view->horizontalScrollBar()->sliderPosition());
}

void PVGuiQt::PVStatsListingWidget::selection_changed()
{
	// Abort the thread if running
	__impl::PVCellWidgetBase::cancel_thread();
	refresh();
}

void PVGuiQt::PVStatsListingWidget::set_refresh_buttons_enabled(bool loading)
{
	for (PVCombCol col(0); col < _stats_panel->columnCount(); col++) {
		for (int row = 0; row < _stats_panel->rowCount(); row++) {
			((__impl::PVCellWidgetBase*)_stats_panel->cellWidget(row, col))
			    ->set_refresh_button_enabled(loading);
		}
	}
}

void PVGuiQt::PVStatsListingWidget::axes_comb_changed(bool async /* = true */)
{
	if (not async) {
		return;
	}
	int old_count = _stats_panel->columnCount();
	int new_count = _listing_view->lib_view().get_column_count();
	int delta = new_count - old_count;
	if (delta > 0) {
		for (PVCombCol col(old_count - 1); col < new_count - 1; col++) {
			_stats_panel->insertColumn(col);
			create_item<__impl::PVUniqueValuesCellWidget>(_row_distinct, col);
			create_item<__impl::PVSumCellWidget>(_row_sum, col);
			create_item<__impl::PVMinCellWidget>(_row_min, col);
			create_item<__impl::PVMaxCellWidget>(_row_max, col);
			create_item<__impl::PVAverageCellWidget>(_row_avg, col);
		}
	} else {
		_stats_panel->setColumnCount(
		    _listing_view->lib_view().get_column_count()); // Widgets gets deleted
	}
	resize_panel();
	for (PVCombCol col(0); col < _stats_panel->columnCount(); col++) {
		for (int row = 0; row < _stats_panel->rowCount(); row++) {
			__impl::PVCellWidgetBase* cell_widget =
			    ((__impl::PVCellWidgetBase*)_stats_panel->cellWidget(row, col));
			cell_widget->update_type_capabilities();
			cell_widget->refresh(true);
		}
		_stats_panel->setColumnWidth(col, _listing_view->columnWidth(col));
	}
}

void PVGuiQt::PVStatsListingWidget::vertical_header_section_clicked(const QPoint&)
{
	if (!_vhead_ctxt_menu) {
		return;
	}

	_vhead_ctxt_menu->exec(QCursor::pos());
}

void PVGuiQt::PVStatsListingWidget::keyPressEvent(QKeyEvent* event)
{
	if (PVWidgets::PVHelpWidget::is_help_key(event->key())) {
		if (help_widget()->isHidden()) {
			help_widget()->popup(this, PVWidgets::PVTextPopupWidget::AlignCenter,
			                     PVWidgets::PVTextPopupWidget::ExpandAll);
		}
		return;
	}
}

/******************************************************************************
 *
 * PVGuiQt::__impl::PVVerticalHeaderView
 *
 *****************************************************************************/
PVGuiQt::__impl::PVVerticalHeaderView::PVVerticalHeaderView(PVStatsListingWidget* parent)
    : QHeaderView(Qt::Vertical, parent)
{
	// These two calls are required since they are done on the headers in
	// QTableView::QTableView
	// instead of in QHeaderView::QHeaderView !
	setSectionsClickable(true);
	setHighlightSections(true);

	// Context menu of the horizontal header
	connect(this, SIGNAL(customContextMenuRequested(const QPoint&)), parent,
	        SLOT(vertical_header_section_clicked(const QPoint&)));
	setContextMenuPolicy(Qt::CustomContextMenu);
}

/******************************************************************************
 *
 * PVGuiQt::__impl::PVCellWidgetBase
 *
 *****************************************************************************/
QMovie* PVGuiQt::__impl::PVCellWidgetBase::_loading_movie = nullptr;
std::thread PVGuiQt::__impl::PVCellWidgetBase::_thread = std::thread();
tbb::task_group_context* PVGuiQt::__impl::PVCellWidgetBase::_ctxt = new tbb::task_group_context();
bool PVGuiQt::__impl::PVCellWidgetBase::_thread_running = false;

PVGuiQt::__impl::PVCellWidgetBase::PVCellWidgetBase(QTableWidget* table,
                                                    Squey::PVView& view,
                                                    QTableWidgetItem* item)
    : _table(table)
    , _view(view)
    , _item(item)
    , _refresh_pixmap(QPixmap::fromImage(QImage(":/refresh_small_grey"))) // FIXME
    , _autorefresh_on_pixmap(QPixmap::fromImage(QImage(":/icon_linked")))
    , _autorefresh_off_pixmap(QPixmap::fromImage(QImage(":/icon_unlinked")))
{
	// connect(table->verticalHeader(), SIGNAL(sectionClicked(int)), this,
	// SLOT(vertical_header_clicked(int)));

	_text = new QLabel();

	_refresh_icon = new QPushButton();
	_refresh_icon->setCursor(QCursor(Qt::PointingHandCursor));
	_refresh_icon->setFlat(true);
	_refresh_icon->setStyleSheet("QPushButton { border: none; }");
	_refresh_icon->setIcon(PVModdedIcon("arrows-rotate"));
	_refresh_icon->setFocusPolicy(Qt::NoFocus);
	_refresh_icon->setToolTip("Refresh");
	connect(_refresh_icon, &QAbstractButton::clicked, this, &PVCellWidgetBase::refresh);

	_loading_label = new PVLoadingLabel(this);
	_loading_label->setMovie(get_movie());
	_loading_label->setStyleSheet("QLabel { padding-right: 4px }");
	_loading_label->setVisible(false);
	_loading_label->setCursor(QCursor(Qt::PointingHandCursor));
	_loading_label->setToolTip("Click to abort");
	connect(_loading_label, SIGNAL(clicked()), this, SLOT(cancel_thread()));

	_autorefresh_icon = new QPushButton();
	_autorefresh_icon->setCursor(QCursor(Qt::PointingHandCursor));
	_autorefresh_icon->setFlat(true);
	_autorefresh_icon->setStyleSheet("QPushButton { border: none; }");
	_autorefresh_icon->setIcon(_autorefresh_off_pixmap);
	_autorefresh_icon->setFocusPolicy(Qt::NoFocus);
	_autorefresh_icon->setToolTip("Toggle auto refresh");
	_autorefresh_icon->setVisible(false); // Disabled before having a better job handling pipeline
	connect(_autorefresh_icon, &QAbstractButton::clicked, this,
	        &PVCellWidgetBase::toggle_auto_refresh);

	connect(this, &PVCellWidgetBase::refresh_impl_finished, this, &PVCellWidgetBase::refreshed);

	_main_layout = new QHBoxLayout();
	_main_layout->setSizeConstraint(QLayout::SetMinimumSize);
	QSizePolicy size_policy(QSizePolicy::Preferred, QSizePolicy::Minimum);
	setSizePolicy(size_policy);
	_main_layout->setSpacing(2);
	_main_layout->setContentsMargins(2, 0, 2, 0);
	_main_layout->addWidget(_text);
	_main_layout->addStretch(1);
	_customizable_layout = new QHBoxLayout();
	_customizable_layout->setSizeConstraint(QLayout::SetMinimumSize);
	_main_layout->addLayout(_customizable_layout);
	_main_layout->addWidget(_refresh_icon);
	_main_layout->addWidget(_loading_label);
	//_main_layout->addWidget(_autorefresh_icon);

	// Context menu
	_ctxt_menu = new QMenu(this);
	auto* copy = new QAction(tr("Copy"), _ctxt_menu);
	connect(copy, &QAction::triggered, this, &PVCellWidgetBase::copy_to_clipboard);
	_ctxt_menu->addAction(copy);
	connect(this, &QWidget::customContextMenuRequested, this,
	        &PVCellWidgetBase::context_menu_requested);
	setContextMenuPolicy(Qt::CustomContextMenu);

	setLayout(_main_layout);
}

void PVGuiQt::__impl::PVCellWidgetBase::context_menu_requested(const QPoint&)
{
	if (_valid) {
		_ctxt_menu->exec(QCursor::pos());
	}
}

void PVGuiQt::__impl::PVCellWidgetBase::copy_to_clipboard()
{
	QByteArray ba;
	ba = _text->text().toLocal8Bit();

	QClipboard* clipboard = QApplication::clipboard();
	auto* mdata = new QMimeData();
	mdata->setData("text/plain", ba);
	clipboard->setMimeData(mdata);
}

PVGuiQt::PVStatsListingWidget* PVGuiQt::__impl::PVCellWidgetBase::get_panel()
{
	return (PVGuiQt::PVStatsListingWidget*)
	    PVCore::get_qobject_parent_of_type<PVGuiQt::PVStatsListingWidget*>(this);
}

typename PVGuiQt::PVStatsListingWidget::PVParams& PVGuiQt::__impl::PVCellWidgetBase::get_params()
{
	PVGuiQt::PVStatsListingWidget* stats_panel = get_panel();
	assert(stats_panel);

	return stats_panel->get_params()[get_real_axis_row()][get_real_axis_col()];
}

QMovie* PVGuiQt::__impl::PVCellWidgetBase::get_movie()
{
	if (_loading_movie == nullptr) {
		_loading_movie = new QMovie(":/squey-loading-animation");
		_loading_movie->setScaledSize(QSize(16, 16));
	}
	return _loading_movie;
}

void PVGuiQt::__impl::PVCellWidgetBase::set_loading(bool loading)
{
	_refresh_icon->setVisible(!loading);
	_loading_label->setVisible(loading);
	if (loading) {
		get_movie()->start();
	} else {
		get_movie()->stop();
	}
	get_panel()->set_refresh_buttons_enabled(loading);
}

void PVGuiQt::__impl::PVCellWidgetBase::cancel_thread()
{
	if (_thread_running) {
		_ctxt->cancel_group_execution();
		if (_thread.joinable()) {
			_thread.join();
		}
		_ctxt->reset();
		_thread_running = false;
	}
}

void PVGuiQt::__impl::PVCellWidgetBase::refresh(bool from_cache /* = false */)
{
	QString cached_value = get_params().cached_value;
	bool auto_refresh = get_params().auto_refresh;

	if (!from_cache) {
		if (cached_value.isEmpty()) {
			if (!_thread_running) {
				set_loading(true);
				if (_thread.joinable()) {
					_thread.join();
				}
				_thread_running = true;
				std::thread th(&PVGuiQt::__impl::PVCellWidgetBase::refresh_impl, this);
				_thread.swap(th);
			}
		} else {
			set_valid(cached_value, auto_refresh);
		}
	} else if (from_cache && !cached_value.isEmpty()) {
		set_valid(cached_value, auto_refresh);
	} else {
		set_invalid();
		get_params().cached_value = QString();
	}
}

void PVGuiQt::__impl::PVCellWidgetBase::auto_refresh()
{
	get_params().cached_value = QString();
	bool auto_refresh = get_params().auto_refresh;
	if (auto_refresh) {
		refresh();
	} else {
		set_invalid();
	}
}

void PVGuiQt::__impl::PVCellWidgetBase::refreshed(QString value)
{
	if (value != "") {
		QString cached_value = value;
		get_params().cached_value = cached_value;
		bool auto_refresh = get_params().auto_refresh;
		set_valid(cached_value, auto_refresh);
	}

	QString cached_value = value;
	get_params().cached_value = cached_value;
	bool auto_refresh = get_params().auto_refresh;
	set_valid(cached_value, auto_refresh);

	set_loading(false);
	_thread_running = false;
}

void PVGuiQt::__impl::PVCellWidgetBase::set_refresh_button_enabled(bool loading)
{
	//_refresh_icon->setCursor(QCursor(loading ? Qt::ArrowCursor :
	// Qt::PointingHandCursor));
	if (_valid) {
		return;
	}
	_refresh_icon->setEnabled(!loading);
}

void PVGuiQt::__impl::PVCellWidgetBase::set_invalid()
{
	_refresh_icon->setEnabled(true);
	_text->setText("");
	_valid = false;
}

void PVGuiQt::__impl::PVCellWidgetBase::set_valid(const QString& value, bool auto_refresh)
{
	_text->setText(value);
	_refresh_icon->setEnabled(false);
	_autorefresh_icon->setIcon(auto_refresh ? _autorefresh_on_pixmap : _autorefresh_off_pixmap);
	_refresh_icon->setVisible(!auto_refresh);
	_item->setBackground(QBrush(Qt::NoBrush));
	_valid = true;
	Q_EMIT cell_refreshed(get_widget_cell_col());
}

void PVGuiQt::__impl::PVCellWidgetBase::vertical_header_clicked(int)
{
	refresh();
}

void PVGuiQt::__impl::PVCellWidgetBase::toggle_auto_refresh()
{
	bool& auto_refresh = get_params().auto_refresh;
	auto_refresh = !auto_refresh;

	_autorefresh_icon->setIcon(auto_refresh ? _autorefresh_on_pixmap : _autorefresh_off_pixmap);
	_refresh_icon->setVisible(!auto_refresh);
}

/******************************************************************************
 *
 * PVGuiQt::__impl::PVUniqueValuesCellWidget
 *
 *****************************************************************************/
PVGuiQt::__impl::PVUniqueValuesCellWidget::PVUniqueValuesCellWidget(QTableWidget* table,
                                                                    Squey::PVView& view,
                                                                    QTableWidgetItem* item)
    : PVCellWidgetBase(table, view, item)
{
	auto* unique_values_dlg_icon = new QPushButton();
	unique_values_dlg_icon->setCursor(QCursor(Qt::PointingHandCursor));
	unique_values_dlg_icon->setFlat(true);
	unique_values_dlg_icon->setStyleSheet("QPushButton { border: none; } "
	                                      "QPushButton:pressed { padding-left : "
	                                      "0px; }");
	unique_values_dlg_icon->setIcon(PVModdedIcon("brackets-curly"));
	unique_values_dlg_icon->setFocusPolicy(Qt::NoFocus);
	unique_values_dlg_icon->setToolTip("Show distinct values");
	connect(unique_values_dlg_icon, &QAbstractButton::clicked, this,
	        &PVUniqueValuesCellWidget::show_unique_values_dlg);
	_customizable_layout->addWidget(unique_values_dlg_icon);
}

void PVGuiQt::__impl::PVUniqueValuesCellWidget::refresh_impl()
{
	const pvcop::db::array& col_in = _view.get_rushnraw_parent().column(get_real_axis_col());
	pvcop::db::array col1_out;
	pvcop::db::array col2_out;

	pvcop::db::algo::distinct(col_in, col1_out, col2_out, _view.get_selection_visible_listing());

	Q_EMIT refresh_impl_finished(
	    QString("%L1").arg(col1_out.size())); // We must go back on the Qt thread to update the GUI
}

void PVGuiQt::__impl::PVUniqueValuesCellWidget::show_unique_values_dlg()
{
	bool empty_sel = _view.get_output_layer().get_selection().is_empty();
	if (not empty_sel) {
		auto container = PVCore::get_qobject_parent_of_type<PVDisplays::PVDisplaysContainer*>(this);
		container->create_view_widget(
		    PVDisplays::display_view_if<PVDisplays::PVDisplayViewDistinctValues>(), &_view,
		    {get_real_axis_col()});
	}
}

/******************************************************************************
 *
 * PVGuiQt::__impl::PVSumCellWidget
 *
 *****************************************************************************/
void PVGuiQt::__impl::PVSumCellWidget::refresh_impl()
{
	const pvcop::db::array& column = _view.get_rushnraw_parent().column(get_real_axis_col());

	const pvcop::db::array& sum =
	    pvcop::db::algo::sum(column, _view.get_selection_visible_listing());

	Q_EMIT refresh_impl_finished(
	    QString::fromStdString(sum.at(0))); // We must go back on the Qt thread to update the GUI
}

static bool is_summable(QString column_type)
{
	return (column_type == "number_float" || column_type == "number_double" ||
			column_type == "number_uint64" || column_type == "number_int64" ||
			column_type == "number_uint32" || column_type == "number_int32" ||
			column_type == "number_uint16" || column_type == "number_int16" ||
			column_type == "number_uint8" || column_type == "number_int8" ||
			column_type == "duration");
}

void PVGuiQt::__impl::PVSumCellWidget::update_type_capabilities()
{
	QString column_type = _view.get_parent<Squey::PVSource>()
	                          .get_format()
	                          .get_axes()
	                          .at(get_real_axis_col())
	                          .get_type();
	_is_summable = is_summable(column_type);
	// FIXME : this should be capabilities, not types names !

	setEnabled(_is_summable);
}

/******************************************************************************
 *
 * PVGuiQt::__impl::PVMinCellWidget
 *
 *****************************************************************************/
void PVGuiQt::__impl::PVMinCellWidget::refresh_impl()
{
	const pvcop::db::array& column = _view.get_rushnraw_parent().column(get_real_axis_col());

	const pvcop::db::array& min_array =
	    pvcop::db::algo::min(column, _view.get_selection_visible_listing());

	std::string min = min_array.size() == 1 ? min_array.at(0) : "";

	Q_EMIT refresh_impl_finished(
	    QString(min.c_str())); // We must go back on the Qt thread to update the GUI
}

/******************************************************************************
 *
 * PVGuiQt::__impl::PVMaxCellWidget
 *
 *****************************************************************************/
void PVGuiQt::__impl::PVMaxCellWidget::refresh_impl()
{
	const pvcop::db::array& column = _view.get_rushnraw_parent().column(get_real_axis_col());

	const pvcop::db::array& max_array =
	    pvcop::db::algo::max(column, _view.get_selection_visible_listing());

	std::string max = max_array.size() == 1 ? max_array.at(0) : "";

	Q_EMIT refresh_impl_finished(
	    QString(max.c_str())); // We must go back on the Qt thread to update the GUI
}

/******************************************************************************
 *
 * PVGuiQt::__impl::PVAverageCellWidget
 *
 *****************************************************************************/
void PVGuiQt::__impl::PVAverageCellWidget::refresh_impl()
{
	const pvcop::db::array& column = _view.get_rushnraw_parent().column(get_real_axis_col());

	const pvcop::db::array& avg =
	    pvcop::db::algo::average(column, _view.get_selection_visible_listing());

	// FIXME : check selection is not empty ?
	// FIXME : add support for proper formatting

	Q_EMIT refresh_impl_finished(
	    QString::fromStdString(avg.at(0))); // We must go back on the Qt thread to update the GUI
}

void PVGuiQt::__impl::PVAverageCellWidget::update_type_capabilities()
{
	QString column_type = _view.get_parent<Squey::PVSource>()
	                          .get_format()
	                          .get_axes()
	                          .at(get_real_axis_col())
	                          .get_type();
	_is_summable = is_summable(column_type);
	// FIXME : this should be capabilities, not types names !

	setEnabled(_is_summable);
}
