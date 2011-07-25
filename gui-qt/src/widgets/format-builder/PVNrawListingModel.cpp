#include <PVNrawListingModel.h>
#include <pvrush/PVNraw.h>

#include <QBrush>

PVInspector::PVNrawListingModel::PVNrawListingModel(QObject* parent):
	QAbstractTableModel(parent),
	_is_consistent(false),
	_col_tosel(0),
	_show_sel(false)
{
}

int PVInspector::PVNrawListingModel::rowCount(const QModelIndex &parent) const
{
	// Cf. QAbstractTableModel's documentation. This is for a table view.
	if (parent.isValid() || !_is_consistent)
		return 0;

	return _nraw->get_number_rows();
}

int PVInspector::PVNrawListingModel::columnCount(const QModelIndex& parent) const
{
	// Same as above
	if (parent.isValid() || !_is_consistent)
		return 0;

	return _nraw->get_number_cols();
}

QVariant PVInspector::PVNrawListingModel::data(const QModelIndex& index, int role) const
{
	switch (role) {
		case Qt::DisplayRole:
			return QVariant(_nraw->get_value(index.row(), index.column()));

		case Qt::BackgroundRole:
		{
			if (_show_sel && index.column() == _col_tosel) {
				// TODO: put this color in something more global (taken from PVListingModel.cpp)
				return QBrush(QColor(130, 100, 25));
			}
			break;
		}
	};

	return QVariant();
}

Qt::ItemFlags PVInspector::PVNrawListingModel::flags(const QModelIndex& /*index*/) const
{
	return Qt::ItemIsEnabled;
}

QVariant PVInspector::PVNrawListingModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	if (orientation != Qt::Horizontal || role != Qt::DisplayRole || !_is_consistent)
		return QAbstractTableModel::headerData(section, orientation, role);
	return _nraw->get_axis_name(section);
}

void PVInspector::PVNrawListingModel::set_nraw(PVRush::PVNraw const& nraw)
{
	_nraw = &nraw;
}

void PVInspector::PVNrawListingModel::set_consistent(bool c)
{
	_is_consistent = c;
	if (c == false) {
		// Data about to be changed !
		emit layoutAboutToBeChanged();
	}
	else {
		// Data has been changed
		emit layoutChanged();
	}
}

bool PVInspector::PVNrawListingModel::is_consistent()
{
	return _is_consistent;
}

void PVInspector::PVNrawListingModel::set_selected_column(PVCol col)
{
	_col_tosel = col;
}

void PVInspector::PVNrawListingModel::sel_visible(bool visible)
{
	_show_sel = visible;
	emit layoutChanged();
}
