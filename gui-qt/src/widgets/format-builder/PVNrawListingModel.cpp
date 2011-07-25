#include <PVNrawListingModel.h>
#include <pvrush/PVNraw.h>

PVInspector::PVNrawListingModel::PVNrawListingModel(QObject* parent):
	QAbstractTableModel(parent),
	_is_consistent(false)
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
	if (role != Qt::DisplayRole)
		return QVariant();

	return QVariant(_nraw->get_value(index.row(), index.column()));
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
