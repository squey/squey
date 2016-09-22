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
} // namespace PVCore

namespace PVWidgets
{

class PVArgumentListModel : public QAbstractTableModel
{
  public:
	explicit PVArgumentListModel(QObject* parent = nullptr);
	explicit PVArgumentListModel(PVCore::PVArgumentList& args, QObject* parent = nullptr);

  public:
	void set_args(PVCore::PVArgumentList& args);

  public:
	int rowCount(const QModelIndex& parent) const override;
	int columnCount(const QModelIndex& parent) const override;
	QVariant data(const QModelIndex& index, int role) const override;
	bool setData(const QModelIndex& index, const QVariant& value, int role) override;
	Qt::ItemFlags flags(const QModelIndex& index) const override;
	QVariant headerData(int section, Qt::Orientation orientation, int role) const override;

  protected:
	PVCore::PVArgumentList* _args;
};
} // namespace PVWidgets

#endif /* PVARGUMENTLISTMODEL_H */
