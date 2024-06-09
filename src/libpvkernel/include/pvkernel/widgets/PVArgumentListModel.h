/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVARGUMENTLISTMODEL_H
#define PVARGUMENTLISTMODEL_H

#include <pvkernel/core/PVArgument.h>
#include <qnamespace.h>
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
