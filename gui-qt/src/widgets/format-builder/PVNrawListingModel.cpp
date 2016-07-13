/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <PVNrawListingModel.h>
#include <pvkernel/rush/PVNraw.h>

#include <QBrush>
#include <QFont>

PVInspector::PVNrawListingModel::PVNrawListingModel(QObject* parent)
    : QAbstractTableModel(parent), _nraw(nullptr), _col_tosel(0), _show_sel(false)
{
}

int PVInspector::PVNrawListingModel::rowCount(const QModelIndex&) const
{
	if (not _nraw) {
		return 0;
	}

	return _nraw->get_row_count();
}

int PVInspector::PVNrawListingModel::columnCount(const QModelIndex&) const
{
	if (not _nraw) {
		return 0;
	}
	return _nraw->get_number_cols();
}

QVariant PVInspector::PVNrawListingModel::data(const QModelIndex& index, int role) const
{
	if (not _nraw) {
		return {};
	}

	bool is_element_valid = _nraw->valid_rows_sel().get_line_fast(index.row());

	switch (role) {
	case Qt::DisplayRole:
		if (is_element_valid) {
			return QString::fromStdString(_nraw->at_string(index.row(), index.column()));
		} else {
			return QString::fromStdString(_inv_elts.at(index.row()));
		}
		break;
	case Qt::BackgroundRole: {
		if (_show_sel && index.column() == _col_tosel) {
			// TODO: put this color in something more global (taken from PVListingModel.cpp)
			return QColor(88, 172, 250);
		}
		if (not is_element_valid) {
			return QColor(250, 197, 205);
		}
		break;
	}
	case (Qt::FontRole): {
		QFont f;
		f.setItalic(not is_element_valid);
		return f;
	}
	};

	return QVariant();
}

Qt::ItemFlags PVInspector::PVNrawListingModel::flags(const QModelIndex& /*index*/) const
{
	return Qt::ItemIsEnabled;
}

QVariant PVInspector::PVNrawListingModel::headerData(int section,
                                                     Qt::Orientation orientation,
                                                     int role) const
{
	if (not _nraw) {
		return {};
	}

	if (orientation != Qt::Horizontal || role != Qt::DisplayRole)
		return QAbstractTableModel::headerData(section, orientation, role);
	return _format.get_axes().at(section).get_name();
}

void PVInspector::PVNrawListingModel::set_nraw(PVRush::PVNraw const& nraw)
{
	if (nraw.get_row_count() == 0) {
		_nraw = nullptr;
	} else {
		_nraw = &nraw;
	}
	Q_EMIT layoutChanged();
}

void PVInspector::PVNrawListingModel::set_selected_column(PVCol col)
{
	_col_tosel = col;
}

void PVInspector::PVNrawListingModel::sel_visible(bool visible)
{
	_show_sel = visible;
	Q_EMIT layoutChanged();
}
