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
	explicit PVSimpleStringListModel(container_type const& values, QObject* parent = nullptr)
	    : PVAbstractTableModel(values.size(), parent), _values(values)
	{
	}

	QString export_line(int row, const QString& /*fsep*/) const override
	{
		auto it = _values.begin();
		std::advance(it, rowIndex(row));
		return QString::number(it->first) + " : " + QString::fromStdString(it->second);
	}

  public:
	QVariant data(QModelIndex const& index, int role = Qt::DisplayRole) const override
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

	QVariant headerData(int section, Qt::Orientation orientation, int role) const override
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
} // namespace PVGuiQt

#endif // __PVQUIQT_PVSIMPLELISTSTRINGMODEL_H__
