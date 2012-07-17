#include <picviz/widgets/PVAD2GRFFListModel.h>
#include <picviz/PVSelRowFilteringFunction.h>
#include <picviz/PVView.h>

class AxesCombinationListModel : public QAbstractListModel
{
public:
	AxesCombinationListModel(Picviz::PVView* view, QObject *parent = 0) :
		QAbstractListModel(parent),
		_view(view)
	{
		for (int i = 0 ; i < rowCount() ; i++) {
			// Add observer on index
		}
	}

	int rowCount(const QModelIndex &parent) const
	{
		if (parent.isValid())
			return 0;

		return rowCount();
	}

	int rowCount() const
	{

		return _view->get_axes_count();
	}

	QVariant data(const QModelIndex &index, int role) const
	{
		if (index.row() < 0 || index.row() >= rowCount())
			return QVariant();

		QString axis_name = _view->get_axes_names_list().at(index.row());
		if (role == Qt::DisplayRole)
			return QVariant(axis_name);

		return QVariant();
	}

	Qt::ItemFlags flags(const QModelIndex &index) const
	{
		if (!index.isValid())
			return QAbstractItemModel::flags(index) | Qt::ItemIsDropEnabled | Qt::ItemIsEditable;

		return QAbstractItemModel::flags(index) | Qt::ItemIsDragEnabled | Qt::ItemIsDropEnabled | Qt::ItemIsEditable;
	}

	bool setData(const QModelIndex &index, const QVariant &value, int role)
	{
		PVLOG_INFO("setData\n");
		if (index.row() >= 0 && index.row() < rowCount()) {
			if (role == Qt::EditRole) {
				_view->set_axis_name(index.row(), value.toString());
				emit dataChanged(index, index);
				return true;
			}
		}
		return false;
	}

	void addRow(QModelIndex model_index, Picviz::PVSelRowFilteringFunction_p rff)
	{
		PVLOG_INFO("addRow\n");
		/*int row = 0;
		if (model_index.isValid()) {
			row = model_index.row();
		}
		insertRow(row);
		if (!model_index.isValid()) {
			model_index = index(0, 0);
		}
		QVariant var;
		var.setValue<void*>(rff.get());
		setData(model_index, var, Qt::UserRole);*/
	}

	bool insertRows(int row, int count, const QModelIndex &parent)
	{
		PVLOG_INFO("insertRows\n");
		/*if (count < 1 || row < 0 || row > rowCount(parent))
			return false;

		beginInsertRows(QModelIndex(), row, row + count - 1);

		for (int r = 0; r < count; ++r)
			_rffs.insert(row, Picviz::PVSelRowFilteringFunction_p());

		endInsertRows();

		return true;*/
	}

	/*bool removeRows(int row, int count, const QModelIndex &parent)
	{
		if (count <= 0 || row < 0 || (row + count) > rowCount(parent))
			return false;

		beginRemoveRows(QModelIndex(), row, row + count - 1);

		for (int r = 0; r < count; ++r)
			_rffs.removeAt(row);

		endRemoveRows();

		return true;
	}*/

private:
	Picviz::PVView* _view;
};
