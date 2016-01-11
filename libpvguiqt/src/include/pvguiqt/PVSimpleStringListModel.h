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

namespace PVGuiQt {

template <class Container>
class PVSimpleStringListModel: public PVAbstractTableModel
{
private:
	// Ensure that container is a container of QString's
	BOOST_STATIC_ASSERT((boost::is_same<typename Container::value_type, QString>::value));

public:
	typedef Container container_type;

public:
	PVSimpleStringListModel(container_type const& values, QObject* parent = NULL):
		PVAbstractTableModel(values.size(), parent),
		_values(values)
	{ }

	QString export_line(int row) const override
	{
		return _values.at(row);
	}

public:
	QVariant data(QModelIndex const& index, int role = Qt::DisplayRole) const
	{
		switch(role) {
			case Qt::DisplayRole:
				return _values.at(rowIndex(index));
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

			return QVariant(QString().setNum(section));
		}

		return QVariant();
	}

int columnCount(QModelIndex const& index = QModelIndex()) const override
{
	return 1;
}

private:
	container_type const& _values;
};

}

#endif // __PVQUIQT_PVSIMPLELISTSTRINGMODEL_H__
