/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVARGUMENTLISTMODEL_H
#define PVARGUMENTLISTMODEL_H

#include <pvkernel/core/PVArgument.h>

#include <QAbstractTableModel>
#include <QVariant>

class QObject;

namespace PVCore
{
class PVArgumentList;
}

namespace PVWidgets
{

class PVArgumentListModel : public QAbstractTableModel
{
  public:
	PVArgumentListModel(QObject* parent = 0);
	PVArgumentListModel(PVCore::PVArgumentList& args, QObject* parent = 0);

  public:
	void set_args(PVCore::PVArgumentList& args);

  public:
	int rowCount(const QModelIndex& parent) const;
	int columnCount(const QModelIndex& parent) const;
	QVariant data(const QModelIndex& index, int role) const;
	bool setData(const QModelIndex& index, const QVariant& value, int role);
	Qt::ItemFlags flags(const QModelIndex& index) const;
	QVariant headerData(int section, Qt::Orientation orientation, int role) const;

  protected:
	PVCore::PVArgumentList* _args;
};
}

#endif /* PVARGUMENTLISTMODEL_H */
