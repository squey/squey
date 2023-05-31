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

#ifndef DATATREEMODEL_H
#define DATATREEMODEL_H

#include <squey/PVSource.h>

#include <QAbstractItemModel>

namespace PVGuiQt
{

class PVHiveDataTreeModel : public QAbstractItemModel
{
	Q_OBJECT

  public:
	explicit PVHiveDataTreeModel(Squey::PVSource& root, QObject* parent = nullptr);
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
	Squey::PVSource& _root;
};
} // namespace PVGuiQt

#endif
