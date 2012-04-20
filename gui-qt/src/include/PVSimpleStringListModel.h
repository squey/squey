#ifndef PVINSPECTOR_PVSIMPLELISTSTRINGMODEL_H
#define PVINSPECTOR_PVSIMPLELISTSTRINGMODEL_H

#include <boost/type_traits/is_same.hpp>
#include <boost/static_assert.hpp>

#include <QString>
#include <QAbstractListModel>

namespace PVInspector {

template <class Container>
class PVSimpleStringListModel: public QAbstractListModel
{
private:
	// Ensure that container is a container of QString's
	BOOST_STATIC_ASSERT((boost::is_same<typename Container::value_type, QString>::value));

public:
	typedef Container container_type;

public:
	PVSimpleStringListModel(container_type const& values, QObject* parent = NULL):
		QAbstractListModel(parent),
		_values(values)
	{ }

public:
	int rowCount(QModelIndex const& parent = QModelIndex()) const
	{
		if (parent.isValid()) {
			return 0;
		}

		return _values.size();
	}

	QVariant data(QModelIndex const& index, int role = Qt::DisplayRole) const
	{
		if (role == Qt::DisplayRole) {
			return QVariant(_values.at(index.row()));
		}
		
		return QVariant();
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

private:
	container_type const& _values;
};

}

#endif
