/**
 * \file PVListingModel.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <QtCore>
#include <QtGui>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVColor.h>

#include <picviz/PVSource.h>
#include <picviz/PVView.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVCallHelper.h>

#include <pvguiqt/PVCustomQtRoles.h>
#include <pvguiqt/PVListingModel.h>

/******************************************************************************
 *
 * PVInspector::PVListingModel::PVListingModel
 *
 *****************************************************************************/
PVGuiQt::PVListingModel::PVListingModel(Picviz::PVView_sp& view, QObject* parent):
	QAbstractTableModel(parent),
	_obs(this),
	_view_valid(true)
{
	//row_header_font = QFont("Convergence-Regular", 6);

	select_brush = QBrush(QColor(255, 240, 200));
	unselect_brush = QBrush(QColor(180, 180, 180));
	vheader_font = QFont(":/Convergence-Regular");
	zombie_font_brush = QBrush(QColor(0, 0, 0));

	PVHive::get().register_actor(view, _actor);
	PVHive::get().register_observer(view, _obs);
	PVHive::get().register_observer(view, [=](Picviz::PVView& v) { return &v.get_axes_combination().get_axes_index_list(); }, _obs_axes_comb);

	_obs.connect_about_to_be_deleted(this, SLOT(view_about_to_be_deleted(PVHive::PVObserverBase*)));
	_obs_axes_comb.connect_refresh(this, SLOT(axes_comb_changed()));
}



/******************************************************************************
 *
 * PVGuiQt::PVListingModel::columnCount
 *
 *****************************************************************************/
int PVGuiQt::PVListingModel::columnCount(const QModelIndex &) const 
{
	if (!_view_valid) {
		return 0;
	}

	return lib_view().get_column_count();
}

/******************************************************************************
 *
 * PVGuiQt::PVListingModel::data
 *
 *****************************************************************************/
QVariant PVGuiQt::PVListingModel::data(const QModelIndex &index, int role) const
{
	assert(_view_valid);

	const PVCol org_col = lib_view().get_original_axis_index(index.column());
	switch (role) {
		case (Qt::DisplayRole):
		{
			bool complete;
			QVariant val = lib_view().get_parent<Picviz::PVSource>()->get_value(index.row(), org_col, &complete);
			if (!complete) {
				_incomplete_fields.push_front(row_col_t(index.row(), index.column()));
			}
			return val;
		}

		case (Qt::TextAlignmentRole):
			return (Qt::AlignLeft + Qt::AlignVCenter);

		case (Qt::BackgroundRole):
		{
			const PVRow r = index.row();
			if (lib_view().get_real_output_selection().get_line(r)) {
				const PVCore::PVHSVColor color = lib_view().get_color_in_output_layer(r);
				return QBrush(color.toQColor());
			} else {
				if (lib_view().get_line_state_in_layer_stack_output_layer(index.row())) {
					/* The event is unselected */
					const PVCore::PVHSVColor color = lib_view().get_color_in_output_layer(r);
					return QBrush(color.toQColor().darker(200));
				} else {
					/* The event is a ZOMBIE */
					return zombie_font_brush;
				}
			}
		}
		case (Qt::ForegroundRole):
		{
			const PVRow r = index.row();
			// Show text in white if this is a zombie event
			if (!lib_view().get_real_output_selection().get_line(r) &&
				!lib_view().get_line_state_in_layer_stack_output_layer(r)) {
				return QBrush(Qt::white);
			}
			return QVariant();
		}
		case (PVCustomQtRoles::Sort):
		{
			QVariant ret(QVariant::fromValue<std::string>(std::move(lib_view().get_parent<Picviz::PVSource>()->get_rushnraw().at_string(index.row(), index.column()))));
			return ret;
		}
		case (Qt::FontRole):
		{
			// Set incomplete fields in italic

			QFont f;
			row_col_t idx;
			bool remove = false;
			for (const row_col_t& idx1 : _incomplete_fields) {
				row_col_t idx2 = row_col_t(index.row(), index.column());
				if (idx1 == idx2) {
					idx = idx1;
					remove = true;
					f.setItalic(true);
					break;
				}
			}

			if (remove) {
				_incomplete_fields.remove(idx);
			}

			return f;
		}
	}
	return QVariant();
}


/******************************************************************************
 *
 * PVGuiQt::PVListingModel::flags
 *
 *****************************************************************************/
Qt::ItemFlags PVGuiQt::PVListingModel::flags(const QModelIndex &/*index*/) const
{
	return (Qt::ItemIsEnabled | Qt::ItemIsSelectable);
}


/******************************************************************************
 *
 * PVGuiQt::PVListingModel::headerData
 *
 *****************************************************************************/
QVariant PVGuiQt::PVListingModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	assert(_view_valid);

	switch (role) {
		case (Qt::DisplayRole):
			if (orientation == Qt::Horizontal) {
				if (section >= 0) {
					QString axis_name(lib_view().get_axis_name(section));
					return QVariant(axis_name);
				}
			}
			else
			if (section >= 0) {
				return section;
			}
		break;
		case (Qt::FontRole):
			if (orientation == Qt::Vertical) {
				if (section >= 0) {
					QFont f(vheader_font);
					if (lib_view().get_real_output_selection().get_line(section)) {
						f.setBold(true);
					}
					return f;
				}
			}
			break;
		case (Qt::TextAlignmentRole):
			if (orientation == Qt::Horizontal) {
				return (Qt::AlignLeft + Qt::AlignTop);
			} else {
				return (Qt::AlignRight + Qt::AlignVCenter);
			}
		/*case (Qt::SizeHintRole):
		{
			return QSize(1, 30);
		}
		break;*/
	}

	return QVariant();
}


/******************************************************************************
 *
 * PVGuiQt::PVListingModel::rowCount
 *
 *****************************************************************************/
int PVGuiQt::PVListingModel::rowCount(const QModelIndex &index) const 
{
	if (index.isValid() || !_view_valid) {
		return 0;
	}

	return lib_view().get_parent<Picviz::PVSource>()->get_row_count();
}

void PVGuiQt::PVListingModel::view_about_to_be_deleted(PVHive::PVObserverBase* /*o*/)
{
	beginResetModel();
	_view_valid = false;
	endResetModel();
}

void PVGuiQt::PVListingModel::axes_comb_changed()
{
	reset();
}
