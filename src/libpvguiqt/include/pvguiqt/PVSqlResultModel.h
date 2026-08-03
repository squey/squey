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

#ifndef __PVGUIQT_PVSQLRESULTMODEL_H__
#define __PVGUIQT_PVSQLRESULTMODEL_H__

#include <pvguiqt/PVAbstractTableModel.h>

#include <squey/PVDuckDBQuery.h>

namespace PVGuiQt
{

/**
 * Table model over a SQL result that is not a selection.
 *
 * Deriving from PVAbstractTableModel rather than from QAbstractTableModel means
 * the result gets the listing's paged scrolling, its selection handling and,
 * through PVListDisplayDlg, export and copy -- none of which is worth
 * reimplementing for a result grid.
 */
class PVSqlResultModel : public PVAbstractTableModel
{
	Q_OBJECT

  public:
	explicit PVSqlResultModel(Squey::PVDuckDBQuery::Table table, QObject* parent = nullptr);

  public:
	QString export_line(int row, const QString& fsep) const override;
	QVariant data(QModelIndex const& index, int role = Qt::DisplayRole) const override;
	QVariant headerData(int section, Qt::Orientation orientation, int role) const override;
	int columnCount(QModelIndex const& parent = QModelIndex()) const override;

  private:
	Squey::PVDuckDBQuery::Table _table;
};

} // namespace PVGuiQt

#endif // __PVGUIQT_PVSQLRESULTMODEL_H__
