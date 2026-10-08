/* * MIT License
 *
 * © Squey, 2026
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

#include <pvguiqt/PVSqlResultModel.h>

#include <utility>

PVGuiQt::PVSqlResultModel::PVSqlResultModel(Squey::PVDuckDBQuery::Table table,
                                            QObject* parent /* = nullptr */)
    : PVAbstractTableModel(int(table.rows.size()), parent), _table(std::move(table))
{
}


QString PVGuiQt::PVSqlResultModel::export_line(int row, const QString& fsep) const
{
	// rowIndex() maps the displayed position onto the underlying row, which is
	// what makes sorting and filtering in the view work.
	const auto& values = _table.rows.at(size_t(rowIndex(row)));

	QString line;
	for (size_t i = 0; i < values.size(); ++i) {
		if (i != 0) {
			line += fsep;
		}
		line += QString::fromStdString(values[i]);
	}
	return line;
}

QVariant PVGuiQt::PVSqlResultModel::data(QModelIndex const& index, int role) const
{
	switch (role) {
	case Qt::DisplayRole: {
		const auto& values = _table.rows.at(size_t(rowIndex(index)));
		const auto col = size_t(index.column());
		return col < values.size() ? QString::fromStdString(values[col]) : QVariant();
	}
	case Qt::BackgroundRole:
		if (is_selected(index)) {
			return _selection_brush;
		}
		break;
	default:
		break;
	}
	return {};
}

QVariant PVGuiQt::PVSqlResultModel::headerData(int section,
                                               Qt::Orientation orientation,
                                               int role) const
{
	if (role != Qt::DisplayRole) {
		return {};
	}
	if (orientation == Qt::Horizontal) {
		return section < int(_table.column_names.size())
		           ? QString::fromStdString(_table.column_names[size_t(section)])
		           : QVariant();
	}
	// Vertical header: the position in the result, not a source row id -- these
	// rows are not rows of the source.
	return section + 1;
}

int PVGuiQt::PVSqlResultModel::columnCount(QModelIndex const& /*parent*/) const
{
	return int(_table.column_names.size());
}
