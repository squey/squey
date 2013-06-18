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

// PVlistColNrawDlg
//

PVGuiQt::PVListUniqStringsDlg::PVListUniqStringsDlg(Picviz::PVView_sp& view, PVCol c, PVRush::PVNraw::unique_values_t& values, QWidget* parent):
	PVListDisplayDlg(new __impl::PVListUniqStringsModel(values), parent), _col(c)
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
	_ctxt_args["exps"].setValue(PVCore::PVPlainTextType(cells.join("\n")));
	_ctxt_args["axis"].setValue(PVCore::PVOriginalAxisIndexType(_col));
	PVCore::PVEnumType e = _ctxt_args["entire"].value<PVCore::PVEnumType>();
	e.set_sel(1);
	_ctxt_args["entire"].setValue(e);

	// Show the layout filter widget
	Picviz::PVLayerFilter_p fclone = lib_filter->clone<Picviz::PVLayerFilter>();
	assert(fclone);
	if (_ctxt_process) {
		_ctxt_process->deleteLater();
	}

	// Creating the PVLayerFilterProcessWidget will save the current args for this filter.
	// Then we can change them !
	_ctxt_process = new PVGuiQt::PVLayerFilterProcessWidget(&lib_view(), _ctxt_args, fclone, _values_view);
	_ctxt_process->show();

	connect(_ctxt_process, SIGNAL(accepted()), this, SLOT(hide()));
}

// Private implementation of PVListColNrawModel
//

PVGuiQt::__impl::PVListUniqStringsModel::PVListUniqStringsModel(PVRush::PVNraw::unique_values_t& values, QWidget* parent):
	QAbstractListModel(parent)
{
	_values.reserve(values.size());
	for (std::string_tbb const& s: values) {
		_values.push_back(std::move(s));
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
	if (role == Qt::DisplayRole) {
		assert((size_t) index.row() < _values.size());
		std::string_tbb const& str = _values[index.row()];
		return QVariant(QString::fromUtf8(str.c_str(), str.size()));
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
