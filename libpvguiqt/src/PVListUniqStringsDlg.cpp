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

	_values_view->horizontalHeader()->show();
	_values_view->horizontalHeader()->setResizeMode(QHeaderView::Interactive);
	connect(_values_view->horizontalHeader(), SIGNAL(sectionResized(int, int, int)), this, SLOT(section_resized(int, int, int)));
	_values_view->setItemDelegateForColumn(1, new __impl::PVListUniqStringsDelegate(this));
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

void PVGuiQt::PVListUniqStringsDlg::showEvent(QShowEvent * event)
{
	//PVLOG_INFO("### PVGuiQt::PVListUniqStringsDlg::showEvent\n");
	PVListDisplayDlg::showEvent(event);
	_resize = true;
	resize_section();
}

void PVGuiQt::PVListUniqStringsDlg::resizeEvent(QResizeEvent* event)
{
	//PVLOG_INFO("### PVGuiQt::PVListUniqStringsDlg::resizeEvent isVisible()=%d\n", isVisible());
	PVListDisplayDlg::resizeEvent(event);
	resize_section();
}

void PVGuiQt::PVListUniqStringsDlg::resize_section()
{
	//PVLOG_INFO("### PVGuiQt::PVListUniqStringsDlg::resize_section _values_view->width() = %d _last_section_size = %d\n", _values_view->width(),_last_section_size);
	_values_view->horizontalHeader()->resizeSection(0, _values_view->width() - _last_section_size);
}

void PVGuiQt::PVListUniqStringsDlg::section_resized(int logicalIndex, int oldSize, int newSize)
{
	//PVLOG_INFO("### PVGuiQt::PVListUniqStringsDlg::section_resized %d %d %d isVisible()=%d\n", logicalIndex, oldSize, newSize,isVisible());
	if (logicalIndex == 1 && _resize) {
		_last_section_size = newSize;
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
	if (role == Qt::DisplayRole) {
		if (orientation == Qt::Horizontal) {
			return QVariant();
		}
		
		return QVariant(QString().setNum(section));
	}

	return QVariant();
}

int PVGuiQt::__impl::PVListUniqStringsModel::columnCount(const QModelIndex& /*index*/) const
{
	return 2;
}


void PVGuiQt::__impl::PVListUniqStringsDelegate::paint(
	QPainter* painter,
	const QStyleOptionViewItem& option,
	const QModelIndex& index) const
{
	assert(index.isValid());

	if (index.column() == 1) {
		size_t occurence_count = index.data(Qt::UserRole).toUInt();

		int progress = (float) occurence_count / get_dialog()->get_selection_count() * 100;

		QStyleOptionProgressBar progressBarOption;
		progressBarOption.rect = option.rect;
		progressBarOption.minimum = 0;
		progressBarOption.maximum = 100;
		progressBarOption.progress = progress;
		progressBarOption.text = QString::number(progress) + "%";
		progressBarOption.textVisible = true;

		QApplication::style()->drawControl(QStyle::CE_ProgressBar,
										&progressBarOption, painter);
	 } else {
		 QStyledItemDelegate::paint(painter, option, index);
	 }

}

PVGuiQt::PVListUniqStringsDlg* PVGuiQt::__impl::PVListUniqStringsDelegate::get_dialog() const
{
	 return static_cast<PVGuiQt::PVListUniqStringsDlg*>(parent());
}
