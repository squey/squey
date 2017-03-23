/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVNraw.h>
#include <pvkernel/rush/PVAxisFormat.h>

#include <pvkernel/widgets/PVAbstractRangePicker.h>
#include <pvkernel/widgets/PVLayerNamingPatternDialog.h>

#include <inendi/PVView.h>

#include <pvguiqt/PVAbstractListStatsDlg.h>
#include <pvguiqt/PVLayerFilterProcessWidget.h>
#include <pvguiqt/PVStatsModel.h>

#include <pvkernel/core/PVLogger.h>
#include <pvkernel/core/inendi_bench.h>
#include <pvkernel/core/PVAlgorithms.h>
#include <pvkernel/core/PVPlainTextType.h>
#include <pvkernel/core/PVOriginalAxisIndexType.h>
#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/core/PVProgressBox.h>

#include <pvcop/db/algo.h>
#include <pvcop/db/types.h>

#include <tbb/blocked_range.h>
#include <tbb/task_scheduler_init.h>

#include <QComboBox>
#include <QGroupBox>
#include <QRadioButton>
#include <QPushButton>
#include <QInputDialog>
#include <QMenu>
#include <QPainter>

/******************************************************************************
 * PVGuiQt::__impl::PVAbstractListStatsRangePicker
 *****************************************************************************/

static inline size_t freq_to_count_min(double value, double count)
{
	// see PVGuiQt::PVAbstractListStatsDlg::select_refresh(bool) for the formula
	if (value == 0.) {
		return 0;
	}
	return ceil((((int)(value * 10.) * 0.001) - 0.0005) * count);
}

static inline size_t freq_to_count_max(double value, double count)
{
	// see PVGuiQt::PVAbstractListStatsDlg::select_refresh(bool) for the formula
	if (value == 0.) {
		return 0;
	}
	return floor((((int)(value * 10.) * 0.001) + 0.0005) * count);
}

static inline double count_to_freq_min(double value, double count)
{
	return trunc(((value / count) * 1000.) + 0.5) / 10.;
}

static inline double count_to_freq_max(double value, double count)
{
	return trunc(((value / count) * 1000.) + 0.5) / 10.;
}

namespace PVGuiQt
{

namespace __impl
{

class PVAbstractListStatsRangePicker : public PVWidgets::PVAbstractRangePicker
{
  public:
	PVAbstractListStatsRangePicker(double relative_min_count,
	                               double relative_max_count,
	                               double absolute_max_count,
	                               QWidget* parent = nullptr)
	    : PVWidgets::PVAbstractRangePicker(relative_min_count, relative_max_count, parent)
	    , _relative_min_count(relative_min_count)
	    , _relative_max_count(relative_max_count)
	    , _absolute_max_count(absolute_max_count)
	{
		update_gradient();
	}

	double convert_to(const double& value) const override
	{
		if (_use_percent_mode) {
			return value / max_count() * 100;
		} else {
			return value;
		}
	}

	double convert_from(const double& value) const override
	{
		if (_use_percent_mode) {
			return value * max_count() / 100;
		} else {
			return value;
		}
	}

	void set_mode_count()
	{
		_use_percent_mode = false;
		_min_spinbox->use_floating_point(false);
		_max_spinbox->use_floating_point(false);

		disconnect_spinboxes_from_ranges();

		get_min_spinbox()->setDecimals(0);
		get_max_spinbox()->setDecimals(0);

		get_min_spinbox()->setSingleStep(1);
		get_max_spinbox()->setSingleStep(1);

		get_min_spinbox()->setSuffix("");
		get_max_spinbox()->setSuffix("");

		set_limits(_relative_min_count, _relative_max_count);
		set_range_min(convert_from(get_range_min()));
		set_range_max(convert_from(get_range_max()));

		connect_spinboxes_to_ranges();
	}

	void set_mode_percent()
	{
		_use_percent_mode = true;
		_min_spinbox->use_floating_point(true);
		_max_spinbox->use_floating_point(true);

		disconnect_spinboxes_from_ranges();

		get_min_spinbox()->setDecimals(1);
		get_max_spinbox()->setDecimals(1);

		get_min_spinbox()->setSingleStep(0.1);
		get_max_spinbox()->setSingleStep(0.1);

		get_min_spinbox()->setSuffix(" %");
		get_max_spinbox()->setSuffix(" %");

		use_absolute_max_count(_use_absolute_max_count);

		connect_spinboxes_to_ranges();
	}

	/**
	 * toggle between linear and logarithmic scales
	 */
	void use_logarithmic_scale(bool use_log)
	{
		double rmin = convert_to(get_range_min());
		double rmax = convert_to(get_range_max());

		_use_logarithmic_scale = use_log;
		update_gradient();

		set_range_max(rmax, true);
		set_range_min(rmin, true);
		update();
	}

	/**
	 * toggle between relative and absolute max count
	 */
	void use_absolute_max_count(bool use_count)
	{
		_use_absolute_max_count = use_count;
		update_gradient();

		double rmin = get_range_min();
		double rmax = get_range_max();

		set_limits(convert_to(_relative_min_count), convert_to(_relative_max_count));

		set_range_max(convert_to(rmax), true);
		set_range_min(convert_to(rmin), true);
		update();
	}

	void update_gradient()
	{
		QLinearGradient gradient;

		QLinearGradient linear_gradient;
		double ratio1, ratio2, ratio3;
		QColor color;

		if (_use_logarithmic_scale &&
		    _relative_min_count != _relative_max_count) { // If only one value, use
			                                              // linear scale to avoid
			                                              // divisions by 0.
			ratio1 = PVCore::log_scale(_relative_min_count, 0., max_count());
			ratio3 = PVCore::log_scale(_relative_max_count, 0., max_count());
			ratio2 = ratio1 + (ratio3 - ratio1) / 2;
		} else {
			ratio1 = _relative_min_count / max_count();
			ratio2 = (_relative_min_count + (_relative_max_count - _relative_min_count) / 2) /
			         max_count();
			ratio3 = _relative_max_count / max_count();
		}

		color = QColor::fromHsv((ratio1) * (0 - 120) + 120, 255, 255);
		gradient.setColorAt(0, color);

		color = QColor::fromHsv((ratio2) * (0 - 120) + 120, 255, 255);
		gradient.setColorAt(0.5, color);

		color = QColor::fromHsv((ratio3) * (0 - 120) + 120, 255, 255);
		gradient.setColorAt(1, color);

		_range_ramp->set_gradient(gradient);
	}

  protected:
	double map_from_spinbox(const double& value) const override
	{
		if (_use_logarithmic_scale) {
			double v = convert_from(value);
			if (_use_percent_mode) {
				v = std::max(convert_from(value - convert_to(_relative_min_count)), 1.0);
			}
			return PVCore::log_scale(v, _relative_min_count, _relative_max_count);
		} else {
			return PVWidgets::PVAbstractRangePicker::map_from_spinbox(convert_from(value));
		}
	}

	double map_to_spinbox(const double& value) const override
	{

		if (_use_logarithmic_scale) {
			return convert_to(
			    PVCore::inv_log_scale(value, _relative_min_count, _relative_max_count));
		} else {
			return convert_to(PVWidgets::PVAbstractRangePicker::map_to_spinbox(value));
		}
	}

  private:
	inline double max_count() const
	{
		return _use_absolute_max_count ? _absolute_max_count : _relative_max_count;
	}

  protected:
	double _relative_min_count;
	double _relative_max_count;
	double _absolute_max_count;
	bool _use_logarithmic_scale = true;
	bool _use_absolute_max_count = true;
	bool _use_percent_mode = false;
};
} // namespace __impl
} // namespace PVGuiQt

/******************************************************************************
 *
 * PVGuiQt::PVAbstractListStatsDlg
 *
 *****************************************************************************/
PVGuiQt::PVAbstractListStatsDlg::PVAbstractListStatsDlg(Inendi::PVView& view,
                                                        PVCol c,
                                                        PVStatsModel* m,
                                                        QWidget* parent /* = nullptr */)
    : PVListDisplayDlg(m, parent), _view(&view), _col(c)
{
	QString search_multiples = "search-multiple";
	Inendi::PVLayerFilter::p_type search_multiple =
	    LIB_CLASS(Inendi::PVLayerFilter)::get().get_class_by_name(search_multiples);
	Inendi::PVLayerFilter::p_type fclone = search_multiple->clone<Inendi::PVLayerFilter>();
	Inendi::PVLayerFilter::hash_menu_function_t const& entries = fclone->get_menu_entries();
	Inendi::PVLayerFilter::hash_menu_function_t::const_iterator it_ent;
	for (it_ent = entries.begin(); it_ent != entries.end(); it_ent++) {
		QAction* act = new QAction(it_ent->key(), _values_view);
		act->setData(QVariant(search_multiples)); // Save the name of the layer
		                                          // filter associated to this
		                                          // action
		_ctxt_menu->addAction(act);

		// RH: a little hack. See comment of _msearch_action_for_layer_creation in
		// .h
		if (act->text() == QString("Search for this value")) {
			_msearch_action_for_layer_creation = act;
		}
	}

	_values_view->horizontalHeader()->setSortIndicator(_sort_section,
	                                                   Qt::SortOrder::AscendingOrder);
	sort_by_column(_sort_section);

	_values_view->horizontalHeader()->show();
	_values_view->verticalHeader()->show();

	_values_view->horizontalHeader()->setStretchLastSection(false);
	_values_view->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
	_values_view->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Fixed);

	_values_view->setAlternatingRowColors(true);
	_values_view->setItemDelegateForColumn(1, new __impl::PVListStringsDelegate(this));

	QString biggest_str = QString().fill('0', QString::number(model().size()).size() + 1);
	_values_view->verticalHeader()->setFixedWidth(QFontMetrics(font()).width(biggest_str));

	// Add content for right click menu on vertical headers
	_hhead_ctxt_menu = new QMenu(this);
	connect(_values_view->horizontalHeader(), SIGNAL(customContextMenuRequested(const QPoint&)),
	        this, SLOT(show_hhead_ctxt_menu(const QPoint&)));

	QActionGroup* act_group_scale = new QActionGroup(this);
	act_group_scale->setExclusive(true);
	connect(act_group_scale, SIGNAL(triggered(QAction*)), this, SLOT(scale_changed(QAction*)));
	_act_toggle_linear = new QAction("Linear scale", act_group_scale);
	_act_toggle_linear->setCheckable(true);
	_act_toggle_log = new QAction("Logarithmic scale", act_group_scale);
	_act_toggle_log->setCheckable(true);
	_hhead_ctxt_menu->addAction(_act_toggle_linear);
	_hhead_ctxt_menu->addAction(_act_toggle_log);
	_hhead_ctxt_menu->addSeparator();

	QActionGroup* act_group_max = new QActionGroup(this);
	act_group_max->setExclusive(true);
	connect(act_group_max, SIGNAL(triggered(QAction*)), this, SLOT(max_changed(QAction*)));
	_act_toggle_absolute = new QAction("Absolute max", act_group_max);
	_act_toggle_absolute->setCheckable(true);
	_act_toggle_absolute->setChecked(true);
	_act_toggle_relative = new QAction("Relative max", act_group_max);
	_act_toggle_relative->setCheckable(true);
	_hhead_ctxt_menu->addAction(_act_toggle_absolute);
	_hhead_ctxt_menu->addAction(_act_toggle_relative);
	_hhead_ctxt_menu->addSeparator();

	// By default, show count and percentage.
	_act_show_count = new QAction("Count", _hhead_ctxt_menu);
	_act_show_count->setCheckable(true);
	_act_show_count->setChecked(true);
	_act_show_scientific_notation = new QAction("Scientific notation", _hhead_ctxt_menu);
	_act_show_scientific_notation->setCheckable(true);
	_act_show_percentage = new QAction("Percentage", _hhead_ctxt_menu);
	_act_show_percentage->setCheckable(true);
	_act_show_percentage->setChecked(true);

	// Give formating information to the model
	model().set_format(ValueFormat::Count, true);
	model().set_format(ValueFormat::Percent, true);
	connect(_act_show_count, &QAction::triggered, [&](bool e) {
		model().set_format(ValueFormat::Count, e);
		update_stats_column_width();
	});
	connect(_act_show_scientific_notation, &QAction::triggered, [&](bool e) {
		model().set_format(ValueFormat::Scientific, e);
		update_stats_column_width();
	});
	connect(_act_show_percentage, &QAction::triggered, [&](bool e) {
		model().set_format(ValueFormat::Percent, e);
		update_stats_column_width();
	});

	_hhead_ctxt_menu->addAction(_act_show_count);
	_hhead_ctxt_menu->addAction(_act_show_scientific_notation);
	_hhead_ctxt_menu->addAction(_act_show_percentage);

	//_values_view->setShowGrid(false);
	//_values_view->setStyleSheet("QTableView::item { border-left: 1px solid grey;
	//}");

	QHBoxLayout* hbox = new QHBoxLayout();

	_select_groupbox->setLayout(hbox);

	QVBoxLayout* vl = new QVBoxLayout();

	hbox->addLayout(vl, 1);

	QRadioButton* r1 = new QRadioButton("by count");
	QRadioButton* r2 = new QRadioButton("by frequency");

	vl->addWidget(r1);
	vl->addWidget(r2);

	connect(r1, SIGNAL(toggled(bool)), this, SLOT(select_set_mode_count(bool)));
	connect(r2, SIGNAL(toggled(bool)), this, SLOT(select_set_mode_frequency(bool)));

	QPushButton* b = new QPushButton("Select");
	connect(b, SIGNAL(clicked(bool)), this, SLOT(select_refresh(bool)));

	vl->addWidget(b);

	_select_picker = new __impl::PVAbstractListStatsRangePicker(
	    model().relative_min_count(), model().relative_max_count(), model().absolute_max_count());
	_select_picker->use_logarithmic_scale(model().use_log_scale());
	hbox->addWidget(_select_picker, 2);

	// set default mode to "count"
	r1->click();

	// propagate the scale mode
	_act_toggle_log->setChecked(model().use_log_scale());

	// Copy values menu
	_copy_values_menu = new QMenu();
	_copy_values_act->setMenu(_copy_values_menu);
	_copy_values_with_count_act = new QAction("with count", this);
	_copy_values_with_count_act->setShortcut(
	    QKeySequence(Qt::ControlModifier + Qt::ShiftModifier + Qt::Key_C));
	_copy_values_menu->addAction(_copy_values_with_count_act);
	_copy_values_without_count_act = new QAction("without count", this);
	_copy_values_without_count_act->setShortcut(QKeySequence::Copy);
	_copy_values_menu->addAction(_copy_values_without_count_act);

	// layer creation actions
	_create_layer_with_values_act = new QAction("Create one layer with those values", _values_view);
	_ctxt_menu->addAction(_create_layer_with_values_act);
	_create_layers_for_values_act = new QAction("Create layers from those values", _values_view);
	_ctxt_menu->addAction(_create_layers_for_values_act);

	connect(_btn_sort, SIGNAL(clicked()), this, SLOT(sort()));

	// Bind the click on header to sort the clicked column
	connect(_values_view->horizontalHeader(), &QHeaderView::sectionClicked,
	        [&](int col) { section_clicked(PVCol(col)); });
	_values_view->horizontalHeader()->setContextMenuPolicy(Qt::CustomContextMenu);
}

void PVGuiQt::PVAbstractListStatsDlg::section_clicked(PVCol col)
{
	// Sort
	sort_by_column(col);
}

void PVGuiQt::PVAbstractListStatsDlg::sort()
{
	Qt::SortOrder order = _values_view->horizontalHeader()->sortIndicatorOrder();
	model().sort(0, order);
	_btn_sort->hide();
}

/******************************************************************************
 * show_hhead_ctxt_menu
 *****************************************************************************/

void PVGuiQt::PVAbstractListStatsDlg::show_hhead_ctxt_menu(const QPoint& pos)
{
	// Show the menu if we click on the second column
	if (_values_view->horizontalHeader()->logicalIndexAt(pos) == 1) {
		//  Menu appear next to the mouse
		_hhead_ctxt_menu->exec(QCursor::pos());
	}
}

bool PVGuiQt::PVAbstractListStatsDlg::process_context_menu(QAction* act)
{
	if (PVListDisplayDlg::process_context_menu(act)) {
		return true;
	}

	if (act == _copy_values_with_count_act) {
		model().set_copy_count(true);
		copy_selected_to_clipboard();
		return true;
	}

	if (act == _copy_values_without_count_act) {
		model().set_copy_count(false);
		copy_selected_to_clipboard();
		return true;
	}

	if (act == _create_layer_with_values_act) {
		create_layer_with_selected_values();
		return true;
	}

	if (act == _create_layers_for_values_act) {
		create_layers_for_selected_values();
		return true;
	}

	if (act) { // TODO : Check it is the correct act?
		QStringList values;

		model().current_selection().visit_selected_lines(
		    [&](int row_id) { values << QString::fromStdString(model().value_col().at(row_id)); });

		multiple_search(act, values);
		return true;
	}

	return false;
}

void PVGuiQt::PVAbstractListStatsDlg::scale_changed(QAction* act)
{
	if (act) {
		bool use_log = (act == _act_toggle_log);
		_select_picker->use_logarithmic_scale(use_log);
		model().set_use_log_scale(use_log);
		_values_view->update();
		_values_view->horizontalHeader()->viewport()->update();
	}
}

void PVGuiQt::PVAbstractListStatsDlg::max_changed(QAction* act)
{
	if (act) {
		model().set_use_absolute(act == _act_toggle_absolute);
		_act_toggle_linear->setChecked(act == _act_toggle_relative);
		_act_toggle_log->setChecked(act == _act_toggle_absolute);
		_select_picker->use_logarithmic_scale(act == _act_toggle_absolute);
		model().set_use_log_scale(act == _act_toggle_absolute);
		_select_picker->use_absolute_max_count(act == _act_toggle_absolute);
		/* as toggling absolute/relative mode changes the largest bounding-box
		 * of scientific and percentage statistics, the statistics column has
		 * to be updated.
		 */
		update_stats_column_width();
		_values_view->update();
		_values_view->horizontalHeader()->viewport()->update();
	}
}

void PVGuiQt::PVAbstractListStatsDlg::select_set_mode_count(bool checked)
{
	if (checked) {
		_select_picker->set_mode_count();
		_select_is_count = true;
	}
}

void PVGuiQt::PVAbstractListStatsDlg::select_set_mode_frequency(bool checked)
{
	if (checked) {
		_select_picker->set_mode_percent();
		_select_is_count = false;
	}
}

void PVGuiQt::PVAbstractListStatsDlg::select_refresh(bool)
{
	/**
	 * As percentage are rounded to be displayed using "%.1f", the entries
	 * can also not be selected using their exact values but using their
	 * rounded ones.
	 *
	 * So that, the nice formula to get the count values corresponding to
	 * the displayed percentage are (in LaTeX):
	 * - v_{min} = \lceil N × ( \frac{ \lfloor 10 × p_{min} \rfloor}{1000} -
	 *\frac{5}{10000} ) \rceil
	 * - v_{max} = \lfloor N × ( \frac{ \lfloor 10 × p_{max} \rfloor}{1000} +
	 *\frac{5}{10000} ) \rfloor
	 * where:
	 * - p_{min} is the lower bound percentage
	 * - p_{max} is the upper bound percentage
	 * - N is the events count
	 */
	double vmin;
	double vmax;
	if (_select_is_count) {
		vmin = _select_picker->get_range_min();
		vmax = _select_picker->get_range_max();
	} else {
		vmin = freq_to_count_min(
		    count_to_freq_min(_select_picker->get_range_min(), model().max_count()),
		    model().max_count());
		vmax = freq_to_count_max(
		    count_to_freq_max(_select_picker->get_range_max(), model().max_count()),
		    model().max_count());
	}

	int row_count = model().stat_col().size();

	BENCH_START(select_values);

	PVCore::PVProgressBox::progress(
	    [this, vmax, vmin](PVCore::PVProgressBox& pbox) {
		    pbox.set_enable_cancel(true);
		    const pvcop::db::array& col2_array = model().stat_col();
		    std::string min_, max_;

		    // Manual check for typing convertion to string.
		    // FIXME : We should handle this with more specific widgets for each
		    // type.
		    switch (col2_array.type()) {
		    case pvcop::db::type_t::type_uint32:
			    min_ = std::to_string(static_cast<uint32_t>(vmin));
			    max_ = std::to_string(static_cast<uint32_t>(vmax));
			    break;
		    case pvcop::db::type_t::type_int32:
			    min_ = std::to_string(static_cast<int32_t>(vmin));
			    max_ = std::to_string(static_cast<int32_t>(vmax));
			    break;
		    case pvcop::db::type_t::type_uint64:
			    min_ = std::to_string(static_cast<uint64_t>(vmin));
			    max_ = std::to_string(static_cast<uint64_t>(vmax));
			    break;
		    case pvcop::db::type_t::type_int64:
			    min_ = std::to_string(static_cast<int64_t>(vmin));
			    max_ = std::to_string(static_cast<int64_t>(vmax));
			    break;
		    case pvcop::db::type_t::type_double:
			    min_ = std::to_string(static_cast<double>(vmin));
			    max_ = std::to_string(static_cast<double>(vmax));
			    break;
		    default:
			    PVLOG_ERROR(("Incorrect type to compute range selection." +
			                 std::to_string(col2_array.type()))
			                    .c_str());
			    return;
		    }
		    model().reset_selection();
		    Inendi::PVSelection& sel = model().current_selection();

		    pvcop::db::algo::range_select(col2_array, min_, max_, pvcop::db::selection(), sel);

		},
	    QObject::tr("Computing selection..."), this);

	// Update the viewport to display selection.
	_values_view->viewport()->update();

	(void)row_count;
	BENCH_END(select_values, "select_values", 0, 0, 1, row_count);
}

void PVGuiQt::PVAbstractListStatsDlg::sort_by_column(PVCol col)
{
	_values_view->horizontalHeader()->setSortIndicatorShown(true);

	PVCol section = (PVCol)_values_view->horizontalHeader()->sortIndicatorSection();

	Qt::SortOrder old_order = _values_view->horizontalHeader()->sortIndicatorOrder();
	Qt::SortOrder new_order = col == _sort_section ? (Qt::SortOrder) not(bool) old_order
	                                               : col == 0 ? Qt::SortOrder::AscendingOrder
	                                                          : Qt::SortOrder::DescendingOrder;

	model().sort(col, new_order);

	_sort_section = section;
}

void PVGuiQt::PVAbstractListStatsDlg::multiple_search(QAction* act,
                                                      const QStringList& sl,
                                                      bool hide_dialog)
{

	// Get the filter associated with that menu entry
	QString filter_name = act->data().toString();
	Inendi::PVLayerFilter_p lib_filter =
	    LIB_CLASS(Inendi::PVLayerFilter)::get().get_class_by_name(filter_name);
	if (!lib_filter) {
		PVLOG_ERROR("(listing context-menu) filter '%s' does not exist !\n",
		            qPrintable(filter_name));
		return;
	}

	Inendi::PVLayerFilter::hash_menu_function_t entries = lib_filter->get_menu_entries();
	QString act_name = act->text();
	if (entries.find(act_name) == entries.end()) {
		PVLOG_ERROR("(listing context-menu) unable to find action '%s' in filter '%s'.\n",
		            qPrintable(act_name), qPrintable(filter_name));
		return;
	}
	Inendi::PVLayerFilter::ctxt_menu_f args_f = entries[act_name];

	// Set the arguments
	_ctxt_args = lib_view()->get_last_args_filter(filter_name);

	PVCore::PVArgumentList custom_args = args_f(0U, (PVCombCol)0, _col, sl.join("\n"));
	PVCore::PVArgumentList_set_common_args_from(_ctxt_args, custom_args);

	// Show the layout filter widget
	Inendi::PVLayerFilter_p fclone = lib_filter->clone<Inendi::PVLayerFilter>();
	assert(fclone);
	if (_ctxt_process) {
		_ctxt_process->deleteLater();
	}

	// Creating the PVLayerFilterProcessWidget will save the current args for this
	// filter.
	// Then we can change them !
	_ctxt_process =
	    new PVGuiQt::PVLayerFilterProcessWidget(lib_view(), _ctxt_args, fclone, _values_view);

	if (hide_dialog) {
		connect(_ctxt_process, SIGNAL(accepted()), this, SLOT(close()));
	}

	if (custom_args.get_edition_flag()) {
		_ctxt_process->show();
	} else {
		_ctxt_process->save_Slot();
	}
}

void PVGuiQt::PVAbstractListStatsDlg::ask_for_copying_count()
{
	if (QMessageBox::question(
	        this, tr("Copy count values"), tr("Do you want to copy count values as well?"),
	        QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes) == QMessageBox::Yes) {
		model().set_copy_count(true);
	} else {
		model().set_copy_count(false);
	}
}

/******************************************************************************
 * PVGuiQt::PVAbstractListStatsDlg::create_layer_with_selected_values
 *****************************************************************************/

void PVGuiQt::PVAbstractListStatsDlg::create_layer_with_selected_values()
{
	PVWidgets::PVLayerNamingPatternDialog dlg("create one new layer with all selected values",
	                                          "string format for new layer's name", "%v",
	                                          PVWidgets::PVLayerNamingPatternDialog::ON_TOP, this);

	if (dlg.exec() == QDialog::Rejected) {
		return;
	}

	QString text = dlg.get_name_pattern();
	PVWidgets::PVLayerNamingPatternDialog::insert_mode mode = dlg.get_insertion_mode();

	Inendi::PVLayerStack& ls = lib_view()->get_layer_stack();

	text.replace("%l", ls.get_selected_layer().get_name());
	text.replace("%a", lib_view()->get_axes_combination().get_axis(_col).get_name());

	QStringList sl;
	QStringList value_names;

	model().current_selection().visit_selected_lines([&](int row_id) {
		QString s = QString::fromStdString(model().value_col().at(row_id));
		sl += s;
		if (s.isEmpty()) {
			value_names += "(empty)";
		} else {
			value_names += s;
		}
	});
	text.replace("%v", value_names.join(","));

	/*
	 * The process is little bit heavy:
	 * - backup the current layer's index
	 * - backup the current selection (the output_layer's one...
	 *   not volatile/floating/whatever)
	 * - run the multiple-search
	 * - create a new layer (do not for forget to notify the layerstackview
	 *   through the hive)
	 * - need to hide the newly created layer (it's better:)
	 * - need to move the newly created layer to the right place
	 * - do a commit from the selection to the newly created layer (like a
	 *   Alt-k and through the hive)
	 * - restore the (old) current layer's index (through the hive)
	 * - set the volatile selection to the backed-up one and force
	 *   reprocessing...
	 */
	int old_selected_layer_index = ls.get_selected_layer_index();
	Inendi::PVSelection old_sel(lib_view()->get_output_layer().get_selection());

	multiple_search(_msearch_action_for_layer_creation, sl, false);

	lib_view()->add_new_layer(text);
	Inendi::PVLayer& layer = lib_view()->get_layer_stack().get_selected_layer();
	int ls_index = lib_view()->get_layer_stack().get_selected_layer_index();
	lib_view()->toggle_layer_stack_layer_n_visible_state(ls_index);

	// We need to configure the layer
	lib_view()->commit_selection_to_layer(layer);
	lib_view()->update_current_layer_min_max();
	lib_view()->compute_selectable_count(layer);

	if (mode != PVWidgets::PVLayerNamingPatternDialog::ON_TOP) {
		int insert_pos;

		if (mode == PVWidgets::PVLayerNamingPatternDialog::ABOVE_CURRENT) {
			insert_pos = old_selected_layer_index + 1;
		} else {
			insert_pos = old_selected_layer_index;
			++old_selected_layer_index;
		}
		lib_view()->move_selected_layer_to(insert_pos);
	}

	ls.set_selected_layer_index(old_selected_layer_index);

	/* as the layer-stack has been changed, force its update at the same time as we restore the
	 * original selection
	 */
	lib_view()->set_selection_view(old_sel, true);
}

/******************************************************************************
 * PVGuiQt::PVAbstractListStatsDlg::create_layers_for_selected_values
 *****************************************************************************/

void PVGuiQt::PVAbstractListStatsDlg::create_layers_for_selected_values()
{
	Inendi::PVLayerStack& ls = lib_view()->get_layer_stack();

	int layer_num = model().current_selection().bit_count();
	int layer_max = INENDI_LAYER_STACK_MAX_DEPTH - ls.get_layer_count();
	if (layer_num >= layer_max) {
		QMessageBox::critical(this, "multiple layer creation",
		                      QString("You try to create %1 layer(s) but no more "
		                              "than %2 layer(s) can be created")
		                          .arg(layer_num)
		                          .arg(layer_max));
		return;
	}

	/* now, we can ask for the layers' names format
	 */
	PVWidgets::PVLayerNamingPatternDialog dlg("create one new layer with all selected values",
	                                          "string format for new layer's name", "%v",
	                                          PVWidgets::PVLayerNamingPatternDialog::ON_TOP, this);

	if (dlg.exec() == QDialog::Rejected) {
		return;
	}

	QString text = dlg.get_name_pattern();
	PVWidgets::PVLayerNamingPatternDialog::insert_mode mode = dlg.get_insertion_mode();

	/* some "static" formatting
	 */
	text.replace("%l", ls.get_selected_layer().get_name());
	text.replace("%a", lib_view()->get_axes_combination().get_axis(_col).get_name());

	/*
	 * The process is little bit heavy:
	 * - backup the current layer's index
	 * - backup the current selection (the output_layer's one...
	 *   not volatile/floating/whatever)
	 * and for each layer to create:
	 * - run the multiple-search
	 * - create a new layer (do not for forget to notify the layerstackview
	 *   through the hive)
	 * - need to hide the newly created layer (it's better:)
	 * - need to move the newly created layer to the right place
	 * - do a commit from the selection to the newly created layer (like a
	 *   Alt-k and through the hive)
	 * - restore the (old) current layer's index (through the hive)
	 * - set the volatile selection to the backed-up one and force
	 *   reprocessing...
	 */

	Inendi::PVSelection old_sel(lib_view()->get_output_layer().get_selection());
	int old_selected_layer_index = ls.get_selected_layer_index();

	/* layers creation
	 */

	int offset = 1;
	model().current_selection().visit_selected_lines([&](int row_id) {
		QString layer_name(text);
		QString s = QString::fromStdString(model().value_col().at(row_id));
		if (s.isEmpty()) {
			layer_name.replace("%v", "(empty)");
		} else {
			layer_name.replace("%v", s);
		}

		QStringList sl;
		sl.append(s);
		// The old selection need to be reset between each value as a LayerFilter may update the
		// selection (and this one do);
		lib_view()->set_selection_view(old_sel);
		multiple_search(_msearch_action_for_layer_creation, sl, false);

		lib_view()->add_new_layer(layer_name);
		Inendi::PVLayer& layer = lib_view()->get_layer_stack().get_selected_layer();
		int ls_index = lib_view()->get_layer_stack().get_selected_layer_index();
		lib_view()->toggle_layer_stack_layer_n_visible_state(ls_index);

		// We need to configure the layer
		lib_view()->commit_selection_to_layer(layer);
		lib_view()->update_current_layer_min_max();
		lib_view()->compute_selectable_count(layer);

		if (mode != PVWidgets::PVLayerNamingPatternDialog::ON_TOP) {
			int insert_pos;

			if (mode == PVWidgets::PVLayerNamingPatternDialog::ABOVE_CURRENT) {
				insert_pos = old_selected_layer_index + offset;
			} else {
				insert_pos = old_selected_layer_index;
				++old_selected_layer_index;
			}
			lib_view()->move_selected_layer_to(insert_pos);
		}
		++offset;
	});

	ls.set_selected_layer_index(old_selected_layer_index);

	/* as the layer-stack has been changed, force its update at the same time as we restore the
	 * original selection
	 */
	lib_view()->set_selection_view(old_sel, true);
}

/******************************************************************************
 *
 * PVGuiQt::__impl::PVListUniqStringsDelegate
 *
 *****************************************************************************/

void PVGuiQt::__impl::PVListStringsDelegate::paint(QPainter* painter,
                                                   const QStyleOptionViewItem& option,
                                                   const QModelIndex& index) const
{
	assert(index.isValid());

	QStyledItemDelegate::paint(painter, option, index);

	if (index.column() == 1) {
		const pvcop::db::array& col2_array = d()->model().stat_col();

		int real_index = d()->model().rowIndex(index);

		if (col2_array.has_invalid() and col2_array.invalid_selection()[real_index]) {
			painter->drawText(option.rect.x(), option.rect.y(), option.rect.width(),
			                  option.rect.height(), Qt::AlignCenter, "N/A");
			return;
		}

		double occurence_count = QString::fromStdString(col2_array.at(real_index)).toDouble();
		double ratio = occurence_count / d()->max_count();
		double log_ratio = PVCore::log_scale(occurence_count, 0., d()->max_count());
		bool log_scale = d()->use_logarithmic_scale();

		// Draw bounding rectangle
		QRect r(option.rect.x(), option.rect.y(), option.rect.width(), option.rect.height());
		QColor base_color = QPalette().color(QPalette::Base);
		QColor alt_color = QPalette().color(QPalette::AlternateBase);
		painter->fillRect(r, index.row() % 2 ? alt_color : base_color);

		// Fill rectangle with color
		painter->fillRect(
		    option.rect.x(), option.rect.y(), option.rect.width() * (log_scale ? log_ratio : ratio),
		    option.rect.height(),
		    QColor::fromHsv((log_scale ? log_ratio : ratio) * (0 - 120) + 120, 255, 255));
		painter->setPen(Qt::black);

		const bool show_count = d()->_act_show_count->isChecked();
		const bool show_scientific = d()->_act_show_scientific_notation->isChecked();
		const bool show_percentage = d()->_act_show_percentage->isChecked();

		if ((not show_count) and (not show_scientific) and (not show_percentage)) {
			// no stats to draw
			return;
		}

		// initializing the text starting horizontal offset
		int x = option.rect.x() + d()->_margin_stats;

		if (show_count) {
			int field_size = d()->_field_size_count;

			painter->drawText(x, option.rect.y(), field_size, option.rect.height(), Qt::AlignRight,
			                  PVStatsModel::format_occurence(occurence_count));
			x += field_size;
		}

		if (show_scientific) {
			int field_size = d()->_field_size_scientific;

			if (show_count) {
				x += d()->_spacing_cs;
			}

			painter->drawText(x, option.rect.y(), field_size, option.rect.height(), Qt::AlignRight,
			                  PVStatsModel::format_scientific_notation(ratio));
			x += field_size;
		}

		if (show_percentage) {
			int field_size = d()->_field_size_percentage;

			if (show_scientific) {
				x += d()->_spacing_sp;
			} else if (show_count) {
				x += d()->_spacing_cp;
			}

			painter->drawText(x, option.rect.y(), field_size, option.rect.height(), Qt::AlignRight,
			                  PVStatsModel::format_percentage(ratio));
		}
	}
}

/******************************************************************************
 *
 * PVGuiQt::__impl::PVListUniqStringsDelegate::d
 *
 *****************************************************************************/

PVGuiQt::PVAbstractListStatsDlg* PVGuiQt::__impl::PVListStringsDelegate::d() const
{
	return static_cast<PVGuiQt::PVAbstractListStatsDlg*>(parent());
}

/******************************************************************************
 *
 * PVAbstractListStatsDlg::update_stats_column_width
 *
 *****************************************************************************/

void PVGuiQt::PVAbstractListStatsDlg::update_stats_column_width()
{
	QFontMetrics fm(_values_view->font());

	const bool show_count = _act_show_count->isChecked();
	const bool show_scientific = _act_show_scientific_notation->isChecked();
	const bool show_percentage = _act_show_percentage->isChecked();

	// initialize the spacing
	_spacing_cp = _spacing_cs = _spacing_sp = stats_default_spacing;

	// compute widths for each statistic
	if (show_count) {
		double v = converting_digits_to_nines_at_given_precision(relative_max_count());
		_field_size_count = fm.width(QLocale::system().toString(v, 'f', 0));
	} else {
		_field_size_count = 0;
	}

	if (show_scientific) {
		double v = converting_digits_to_nines_at_given_precision(relative_max_count() / max_count(),
		                                                         0.001);
		_field_size_scientific = fm.width(PVStatsModel::format_scientific_notation(v));
	} else {
		_field_size_scientific = 0;
	}

	if (show_percentage) {
		double v = converting_digits_to_nines_at_given_precision(relative_max_count() / max_count(),
		                                                         0.001);
		_field_size_percentage = fm.width(PVStatsModel::format_percentage(v));
	} else {
		_field_size_percentage = 0;
	}

	int column_width = _field_size_count + _field_size_scientific + _field_size_percentage;

	// compute spacing between statistics
	if (show_count and show_scientific and show_percentage) {
		column_width += _spacing_cs + _spacing_sp;
	} else if (show_count and show_scientific) {
		column_width += _spacing_cs;
	} else if (show_count and show_percentage) {
		column_width += _spacing_cp;
	} else if (show_scientific and show_percentage) {
		column_width += _spacing_sp;
	}

	// compute margin size
	if (show_count or show_scientific or show_percentage) {
		// at least one statistic, make sure the stats column is not too small
		if (column_width > stats_minimal_column_width) {
			_margin_stats = stats_default_margin;
		} else {
			_margin_stats = std::max((stats_minimal_column_width - column_width) / 2,
			                         (int)stats_default_margin);
		}

		column_width += 2 * _margin_stats;
	} else {
		// no shown stats, force a minimal width for the stat column
		_margin_stats = 0;
		column_width = stats_minimal_column_width;
	}

	_values_view->horizontalHeader()->resizeSection(1, column_width);
}

/******************************************************************************
 *
 * PVAbstractListStatsDlg::keyPressEvent
 *
 *****************************************************************************/
void PVGuiQt::PVAbstractListStatsDlg::keyPressEvent(QKeyEvent* event)
{
	if (event->matches(QKeySequence::Copy)) {
		_values_view->table_model()->commit_selection();
		model().set_copy_count(false);
		copy_selected_to_clipboard();
	} else if ((event->modifiers() & Qt::ControlModifier) &&
	           (event->modifiers() & Qt::ShiftModifier) && event->key() == Qt::Key_C) {
		_values_view->table_model()->commit_selection();
		model().set_copy_count(true);
		copy_selected_to_clipboard();
	} else {
		PVListDisplayDlg::keyPressEvent(event);
	}
}

/******************************************************************************
 *
 * PVAbstractListStatsDlg::resizeEvent
 *
 *****************************************************************************/

void PVGuiQt::PVAbstractListStatsDlg::resizeEvent(QResizeEvent* event)
{
	PVListDisplayDlg::resizeEvent(event);
	update_stats_column_width();
}
