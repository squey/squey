/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef DATATREEMODEL_H
#define DATATREEMODEL_H

#include <inendi/PVSource.h>

#include <QAbstractItemModel>

namespace PVGuiQt
{

class PVHiveDataTreeModel : public QAbstractItemModel
{
	Q_OBJECT

  public:
	explicit PVHiveDataTreeModel(Inendi::PVSource& root, QObject* parent = nullptr);
	QModelIndex index(int row, int column, const QModelIndex& parent) const override;

	int pos_from_obj(PVCore::PVDataTreeObject const* o) const;

  protected:
	int rowCount(const QModelIndex& index) const override;
	int columnCount(const QModelIndex&) const override { return 1; }

	QVariant data(const QModelIndex& index, int role) const override;

	Qt::ItemFlags flags(const QModelIndex& /*index*/) const override
	{
		return Qt::ItemIsSelectable | Qt::ItemIsEnabled | Qt::ItemIsUserCheckable;
	}

	QModelIndex parent(const QModelIndex& index) const override;

  private Q_SLOTS:
	void update_obj(const PVCore::PVDataTreeObject* obj_base);

  private:
	Inendi::PVSource& _root;
};
} // namespace PVGuiQt

#endif
