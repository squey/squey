/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef __PVQUIQT_PVSIMPLELISTSTRINGMODEL_H__
#define __PVQUIQT_PVSIMPLELISTSTRINGMODEL_H__

#include <boost/type_traits/is_same.hpp>
#include <boost/static_assert.hpp>

#include <QString>
#include <pvguiqt/PVAbstractTableModel.h>

namespace PVGuiQt
{

class PVSimpleStringListModel : public PVAbstractTableModel
{
  public:
	using container_type = std::map<size_t, std::string>;

  public:
	PVSimpleStringListModel(container_type const& values, QObject* parent = nullptr)
	    : PVAbstractTableModel(values.size(), parent), _values(values)
	{
	}

	QString export_line(int row) const override
	{
		auto it = _values.begin();
		std::advance(it, rowIndex(row));
		return QString::number(it->first) + " : " + QString::fromStdString(it->second);
	}

  public:
	QVariant data(QModelIndex const& index, int role = Qt::DisplayRole) const
	{
		switch (role) {
		case Qt::DisplayRole: {
			auto it = _values.begin();
			std::advance(it, rowIndex(index));
			return QString::fromStdString(it->second);
		}
		case Qt::BackgroundRole:
			if (is_selected(index)) {
				return _selection_brush;
			}
		}

		return {};
	}

	QVariant headerData(int section, Qt::Orientation orientation, int role) const
	{
		if (role == Qt::DisplayRole) {
			if (orientation == Qt::Horizontal) {
				return QVariant();
			}

			auto it = _values.begin();
			std::advance(it, rowIndex(section));
			return QString().setNum(it->first);
		}

		return QVariant();
	}

	int columnCount(QModelIndex const&) const override { return 1; }

  private:
	container_type const& _values;
};
}

#endif // __PVQUIQT_PVSIMPLELISTSTRINGMODEL_H__
