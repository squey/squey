/**
 * \file PVListUniqStrings.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVPlainTextType.h>
#include <pvkernel/core/PVOriginalAxisIndexType.h>
#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/rush/PVNraw.h>

#include <pvguiqt/PVListUniqStringsDlg.h>
#include <pvguiqt/PVLayerFilterProcessWidget.h>

#include <pvkernel/core/PVLogger.h>
#include <pvguiqt/PVStringSortProxyModel.h>

// PVlistColNrawDlg
//

PVGuiQt::PVListUniqStringsDlg::PVListUniqStringsDlg(
	Picviz::PVView_sp& view,
	PVCol c,
	PVRush::PVNraw::unique_values_t& values,
	size_t selection_count,
	QWidget* parent
) :
	PVListDisplayDlg(new __impl::PVListUniqStringsModel(values), parent), _col(c), _selection_count(selection_count)
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
	_values_view->horizontalHeader()->setResizeMode(QHeaderView::Interactive);
	connect(_values_view->horizontalHeader(), SIGNAL(sectionResized(int, int, int)), this, SLOT(section_resized(int, int, int)));
	_values_view->setItemDelegateForColumn(1, new __impl::PVListUniqStringsDelegate(this));

	QActionGroup* act_group = new QActionGroup(this);
	act_group->setExclusive(true);
	_act_toggle_linear = new QAction("Linear scale", act_group);
	_act_toggle_linear->setCheckable(true);
	_act_toggle_linear->setChecked(!_use_logorithmic_scale);
	_act_toggle_log = new QAction("Logarithmic scale", act_group);
	_act_toggle_log->setCheckable(true);
	_act_toggle_log->setChecked(_use_logorithmic_scale);
	_hhead_ctxt_menu->addAction(_act_toggle_linear);
	_hhead_ctxt_menu->addAction(_act_toggle_log);

	//_values_view->setShowGrid(false);
	//_values_view->setStyleSheet("QTableView::item { border-left: 1px solid grey; }");
}

PVGuiQt::PVListUniqStringsDlg::~PVListUniqStringsDlg()
{
	// Force deletion so that the internal std::vector is destroyed!
	model()->deleteLater();
}

void PVGuiQt::PVListUniqStringsDlg::process_context_menu(QAction* act)
{
	PVListDisplayDlg::process_context_menu(act);
	if (act) {
		multiple_search(act);
	}
}

void PVGuiQt::PVListUniqStringsDlg::process_hhead_context_menu(QAction* act)
{
	PVListDisplayDlg::process_hhead_context_menu(act);

	if (act) {
		_use_logorithmic_scale = (act == _act_toggle_log);
		_values_view->update();
	}
}

void PVGuiQt::PVListUniqStringsDlg::showEvent(QShowEvent * event)
{
	PVListDisplayDlg::showEvent(event);
	resize_section();
}

void PVGuiQt::PVListUniqStringsDlg::view_resized()
{
	// We don't want the resize of the view to change the stored last section width
	_store_last_section_width = false;
	resize_section();
}

void PVGuiQt::PVListUniqStringsDlg::resize_section()
{
	_values_view->horizontalHeader()->resizeSection(0, _values_view->width() - _last_section_width);
}

void PVGuiQt::PVListUniqStringsDlg::section_resized(int logicalIndex, int /*oldSize*/, int newSize)
{
	if (logicalIndex == 1) {
		if (_store_last_section_width) {
			_last_section_width = newSize;
		}
		_store_last_section_width = true;
	}
}

void PVGuiQt::PVListUniqStringsDlg::sort_by_column(int col)
{
	PVListDisplayDlg::sort_by_column(col);

	if (col == 1) {
		Qt::SortOrder order =  (Qt::SortOrder)!((bool)_values_view->horizontalHeader()->sortIndicatorOrder());
		proxy_model()->sort(col, order);
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVListUniqStringsDlg::multiple_search
 *
 *****************************************************************************/
void PVGuiQt::PVListUniqStringsDlg::multiple_search(QAction* act)
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

// Private implementation of PVListColNrawModel
//

PVGuiQt::__impl::PVListUniqStringsModel::PVListUniqStringsModel(PVRush::PVNraw::unique_values_t& values, QWidget* parent):
	QAbstractListModel(parent)
{
	for (auto& v : values) {
		_values.emplace_back(std::move(v.first), v.second);
	}
}

int PVGuiQt::__impl::PVListUniqStringsModel::rowCount(QModelIndex const& parent) const
{
	if (parent.isValid()) {
		return 0;
	}

	return _values.size();
}

QVariant PVGuiQt::__impl::PVListUniqStringsModel::data(QModelIndex const& index, int role) const
{
	assert((size_t) index.row() < _values.size());

	if (role == Qt::DisplayRole) {
		switch (index.column()) {
			case 0:
			{
				std::string_tbb const& str = _values[index.row()].first;
				return QVariant(QString::fromUtf8(str.c_str(), str.size()));
			}
			break;
		}
	}
	else if (role == Qt::UserRole) {
		switch (index.column()) {
			case 1:
			{
				return QVariant::fromValue(_values[index.row()].second);
			}
		}
	}

	return QVariant();
}

QVariant PVGuiQt::__impl::PVListUniqStringsModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	QHash<size_t, QString> h;
	h[0] = "Value";
	h[1] = "Frequency";

	if (role == Qt::DisplayRole) {
		if (orientation == Qt::Horizontal) {
			return h[section];
		}
		return QVariant(QString().setNum(section));
	}
	else if (role == Qt::TextAlignmentRole) {
		if (orientation == Qt::Horizontal) {
			return (Qt::AlignLeft + Qt::AlignVCenter);
		}
		else {
			return (Qt::AlignRight + Qt::AlignVCenter);
		}
	}

	return QVariant();
}

int PVGuiQt::__impl::PVListUniqStringsModel::columnCount(const QModelIndex& /*index*/) const
{
	return 2;
}

#define ALTERNATING_BG_COLOR 1

void PVGuiQt::__impl::PVListUniqStringsDelegate::paint(
	QPainter* painter,
	const QStyleOptionViewItem& option,
	const QModelIndex& index) const
{
	assert(index.isValid());

	QStyledItemDelegate::paint(painter, option, index);

	if (index.column() == 1) {
		size_t occurence_count = index.data(Qt::UserRole).toUInt();

		double ratio = (double) occurence_count / get_dialog()->get_selection_count();
		double log_ratio = (double) log(occurence_count) / log(get_dialog()->get_selection_count());
		bool log_scale = get_dialog()->use_logorithmic_scale();

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

		// Draw percentage
		QString percent = QString::number(ratio * 100, 'f', 2) + " %";
		painter->setPen(Qt::black);
		painter->drawText(
			option.rect.x()+thickness,
			option.rect.y()+2*thickness,
			option.rect.width()-thickness,
			option.rect.height()-thickness,
			Qt::AlignCenter,
			percent
		);

		/*QStyleOptionProgressBar progressBarOption;
		progressBarOption.rect = option.rect;
		progressBarOption.minimum = 0;
		progressBarOption.maximum = 100;
		progressBarOption.progress = percentage;
		progressBarOption.text = QString::number(percentage) + "%";
		progressBarOption.textVisible = true;
		QApplication::style()->drawControl(QStyle::CE_ProgressBar, &progressBarOption, painter);*/

	 } else {
		 QStyledItemDelegate::paint(painter, option, index);
	 }

}

PVGuiQt::PVListUniqStringsDlg* PVGuiQt::__impl::PVListUniqStringsDelegate::get_dialog() const
{
	 return static_cast<PVGuiQt::PVListUniqStringsDlg*>(parent());
}
