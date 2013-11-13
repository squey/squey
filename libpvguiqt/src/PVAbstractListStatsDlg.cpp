/**
 * \file PVAbstractListStatsDlg.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

#include <pvkernel/core/PVPlainTextType.h>
#include <pvkernel/core/PVOriginalAxisIndexType.h>
#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/rush/PVNraw.h>

#include <pvguiqt/PVAbstractListStatsDlg.h>
#include <pvguiqt/PVLayerFilterProcessWidget.h>

#include <pvkernel/core/PVLogger.h>
#include <pvguiqt/PVStringSortProxyModel.h>

/******************************************************************************
 *
 * PVGuiQt::PVAbstractListStatsDlg
 *
 *****************************************************************************/
void PVGuiQt::PVAbstractListStatsDlg::init(Picviz::PVView_sp& view)
{
	PVHive::get().register_observer(view, _obs);
	PVHive::get().register_actor(view, _actor);
	_obs.connect_about_to_be_deleted(this, SLOT(deleteLater()));

	QString search_multiples = "search-multiple";
	Picviz::PVLayerFilter::p_type search_multiple = LIB_CLASS(Picviz::PVLayerFilter)::get().get_class_by_name(search_multiples);
	Picviz::PVLayerFilter::p_type fclone = search_multiple->clone<Picviz::PVLayerFilter>();
	Picviz::PVLayerFilter::hash_menu_function_t const& entries = fclone->get_menu_entries();
	Picviz::PVLayerFilter::hash_menu_function_t::const_iterator it_ent;
	for (it_ent = entries.begin(); it_ent != entries.end(); it_ent++) {
		QAction* act = new QAction(it_ent.key(), _values_view);
		act->setData(QVariant(search_multiples)); // Save the name of the layer filter associated to this action
		_ctxt_menu->addAction(act);
	}

	__impl::PVTableViewResizeEventFilter* table_view_resize_event_handler = new __impl::PVTableViewResizeEventFilter();
	_values_view->installEventFilter(table_view_resize_event_handler);
	connect(table_view_resize_event_handler, SIGNAL(resized()), this, SLOT(view_resized()));
	_values_view->horizontalHeader()->show();
	_values_view->verticalHeader()->show();
	_values_view->horizontalHeader()->setResizeMode(QHeaderView::Interactive);
	connect(_values_view->horizontalHeader(), SIGNAL(sectionResized(int, int, int)), this, SLOT(section_resized(int, int, int)));
	_values_view->setItemDelegateForColumn(1, new __impl::PVListStringsDelegate(this));

	QActionGroup* act_group = new QActionGroup(this);
	act_group->setExclusive(true);
	connect(act_group, SIGNAL(triggered(QAction*)), this, SLOT(scale_changed(QAction*)));
	_act_toggle_linear = new QAction("Linear scale", act_group);
	_act_toggle_linear->setCheckable(true);
	_act_toggle_linear->setChecked(!_use_logarithmic_scale);
	_act_toggle_log = new QAction("Logarithmic scale", act_group);
	_act_toggle_log->setCheckable(true);
	_act_toggle_log->setChecked(_use_logarithmic_scale);
	_hhead_ctxt_menu->addAction(_act_toggle_linear);
	_hhead_ctxt_menu->addAction(_act_toggle_log);
	_hhead_ctxt_menu->addSeparator();

	_act_show_count = new QAction("Count", _hhead_ctxt_menu);
	_act_show_count->setCheckable(true);
	_act_show_scientific_notation = new QAction("Scientific notation", _hhead_ctxt_menu);
	_act_show_scientific_notation->setCheckable(true);
	_act_show_percentage = new QAction("Percentage", _hhead_ctxt_menu);
	_act_show_percentage->setCheckable(true);
	_act_show_percentage->setChecked(true);

	_hhead_ctxt_menu->addAction(_act_show_count);
	_hhead_ctxt_menu->addAction(_act_show_scientific_notation);
	_hhead_ctxt_menu->addAction(_act_show_percentage);

	//_values_view->setShowGrid(false);
	//_values_view->setStyleSheet("QTableView::item { border-left: 1px solid grey; }");
}

PVGuiQt::PVAbstractListStatsDlg::~PVAbstractListStatsDlg()
{
	// Force deletion so that the internal std::vector is destroyed!
	model()->deleteLater();
}

bool PVGuiQt::PVAbstractListStatsDlg::process_context_menu(QAction* act)
{
	bool accepted = PVListDisplayDlg::process_context_menu(act);
	if (!accepted && act) {
		multiple_search(act);
		return true;
	}
	return false;
}

void PVGuiQt::PVAbstractListStatsDlg::process_hhead_context_menu(QAction* act)
{
	PVListDisplayDlg::process_hhead_context_menu(act);
}

void PVGuiQt::PVAbstractListStatsDlg::scale_changed(QAction* act)
{
	if (act) {
		_use_logarithmic_scale = (act == _act_toggle_log);
		_values_view->update();
	}
}

void PVGuiQt::PVAbstractListStatsDlg::showEvent(QShowEvent * event)
{
	PVListDisplayDlg::showEvent(event);
	resize_section();
}

void PVGuiQt::PVAbstractListStatsDlg::view_resized()
{
	// We don't want the resize of the view to change the stored last section width
	_store_last_section_width = false;
	resize_section();
}

void PVGuiQt::PVAbstractListStatsDlg::resize_section()
{
	_values_view->horizontalHeader()->resizeSection(0, _values_view->width() - _last_section_width);
}

void PVGuiQt::PVAbstractListStatsDlg::section_resized(int logicalIndex, int /*oldSize*/, int newSize)
{
	if (logicalIndex == 1) {
		if (_store_last_section_width) {
			_last_section_width = newSize;
		}
		_store_last_section_width = true;
	}
}

void PVGuiQt::PVAbstractListStatsDlg::sort_by_column(int col)
{
	PVListDisplayDlg::sort_by_column(col);

	if (col == 1) {
		Qt::SortOrder order =  (Qt::SortOrder)!((bool)_values_view->horizontalHeader()->sortIndicatorOrder());
		proxy_model()->sort(col, order);
	}
}

void PVGuiQt::PVAbstractListStatsDlg::multiple_search(QAction* act)
{

	// Get the filter associated with that menu entry
	QString filter_name = act->data().toString();
	Picviz::PVLayerFilter_p lib_filter = LIB_CLASS(Picviz::PVLayerFilter)::get().get_class_by_name(filter_name);
	if (!lib_filter) {
		PVLOG_ERROR("(listing context-menu) filter '%s' does not exist !\n", qPrintable(filter_name));
		return;
	}

	Picviz::PVLayerFilter::hash_menu_function_t entries = lib_filter->get_menu_entries();
	QString act_name = act->text();
	if (entries.find(act_name) == entries.end()) {
		PVLOG_ERROR("(listing context-menu) unable to find action '%s' in filter '%s'.\n", qPrintable(act_name), qPrintable(filter_name));
		return;
	}
	Picviz::PVLayerFilter::ctxt_menu_f args_f = entries[act_name];

	// Set the arguments
	_ctxt_args = lib_view().get_last_args_filter(filter_name);

	QItemSelectionModel* selection_model = _values_view->selectionModel();
	assert(selection_model);
	QModelIndexList list = selection_model->selection().indexes();
	QStringList cells;
	for (const auto& cell : list) {
		cells.append(cell.data().toString());
	}
	PVCore::PVArgumentList custom_args = args_f(0U, 0, _col, cells.join("\n"));
	PVCore::PVArgumentList_set_common_args_from(_ctxt_args, custom_args);

	// Show the layout filter widget
	Picviz::PVLayerFilter_p fclone = lib_filter->clone<Picviz::PVLayerFilter>();
	assert(fclone);
	if (_ctxt_process) {
		_ctxt_process->deleteLater();
	}

	// Creating the PVLayerFilterProcessWidget will save the current args for this filter.
	// Then we can change them !
	_ctxt_process = new PVGuiQt::PVLayerFilterProcessWidget(&lib_view(), _ctxt_args, fclone, _values_view);
	connect(_ctxt_process, SIGNAL(accepted()), this, SLOT(hide()));

	if (custom_args.get_edition_flag()) {
		_ctxt_process->show();
	} else {
		_ctxt_process->save_Slot();
	}
}


/******************************************************************************
 *
 * PVGuiQt::__impl::PVListUniqStringsDelegate
 *
 *****************************************************************************/

#define ALTERNATING_BG_COLOR 1

void PVGuiQt::__impl::PVListStringsDelegate::paint(
	QPainter* painter,
	const QStyleOptionViewItem& option,
	const QModelIndex& index) const
{
	assert(index.isValid());

	QStyledItemDelegate::paint(painter, option, index);

	if (index.column() == 1) {
		size_t occurence_count = index.data(Qt::UserRole).toUInt();

		double ratio = (double) occurence_count / d()->get_total_count();
		double log_ratio = d()->get_total_count() == 1 ? 1.0 : (double) log(occurence_count) / log(d()->get_total_count());
		bool log_scale = d()->use_logorithmic_scale();

		// Draw bounding rectangle
		size_t thickness = 1;
		QRect r(option.rect.x()/*+2*/, option.rect.y()+thickness, option.rect.width(), option.rect.height()-thickness);
		QColor color("#F2F2F2");
#if ALTERNATING_BG_COLOR
		QColor alt_color("#FBFBFB");
		painter->fillRect(r, index.row() % 2 ? color : alt_color);
#else
		painter->setPen(color); painter->drawRect(r);
#endif

		// Fill rectangle with color
		painter->fillRect(
			option.rect.x()+thickness/*+2*/,
			option.rect.y()+2*thickness,
			option.rect.width()*(log_scale ? log_ratio : ratio)-thickness,
			option.rect.height()-2*thickness,
			QColor::fromHsv((log_scale ? log_ratio : ratio) * (0 - 120) + 120, 255, 255)
		);
		painter->setPen(Qt::black);

		// Draw occurence count and/or scientific notation and/or percentage
		size_t occurence_max_width = 0;
		size_t scientific_notation_max_width = 0;
		size_t percentage_max_width = 0;

		size_t margin = option.rect.width();

		QString occurence;
		QString scientific_notation;
		QString percentage;

		size_t representation_count = 0;
		if (d()->_act_show_count->isChecked()) {
			occurence = format_occurence(occurence_count);
			occurence_max_width = QFontMetrics(painter->font()).width(format_occurence(d()->_max_e));
			margin -= occurence_max_width;
			representation_count++;
		}
		if (d()->_act_show_scientific_notation->isChecked()) {
			scientific_notation = format_scientific_notation(ratio);
			scientific_notation_max_width = QFontMetrics(painter->font()).width(format_scientific_notation(0.27));
			margin -= scientific_notation_max_width;
			representation_count++;
		}
		if (d()->_act_show_percentage->isChecked()) {
			percentage = format_percentage(ratio);
			percentage_max_width = QFontMetrics(painter->font()).width(format_percentage((double)d()->_max_e / d()->get_total_count()));
			margin -= percentage_max_width;
			representation_count++;
		}

		margin /= representation_count+1;

		int x =  option.rect.x()+thickness;
		if (d()->_act_show_count->isChecked()) {
			x += margin;
			painter->drawText(
				x,
				option.rect.y()+2*thickness,
				occurence_max_width,
				option.rect.height()-thickness,
				Qt::AlignRight,
				occurence
			);
			x += occurence_max_width;
		}
		if (d()->_act_show_scientific_notation->isChecked()) {
			x += margin;
			painter->drawText(
				x,
				option.rect.y()+2*thickness,
				scientific_notation_max_width,
				option.rect.height()-thickness,
				Qt::AlignLeft,
				scientific_notation
			);
			x += scientific_notation_max_width;
		}
		if (d()->_act_show_percentage->isChecked()) {
			x += margin;
			painter->drawText(
				x,
				option.rect.y()+2*thickness,
				percentage_max_width,
				option.rect.height()-thickness,
				Qt::AlignRight,
				percentage
			);
		}
	}
	else {
		 QStyledItemDelegate::paint(painter, option, index);
	}
}

PVGuiQt::PVAbstractListStatsDlg* PVGuiQt::__impl::PVListStringsDelegate::d() const
{
	 return static_cast<PVGuiQt::PVAbstractListStatsDlg*>(parent());
}
