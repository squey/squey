//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <PVNrawListingModel.h>
#include <pvkernel/rush/PVNraw.h>

#include <QBrush>
#include <QFont>
#include <QFontMetrics>

PVInspector::PVNrawListingModel::PVNrawListingModel(QObject* parent)
    : QAbstractTableModel(parent), _nraw(nullptr), _col_tosel(0), _show_sel(false)
{
}

int PVInspector::PVNrawListingModel::rowCount(const QModelIndex&) const
{
	if (not _nraw) {
		return 0;
	}

	return _nraw->row_count();
}

int PVInspector::PVNrawListingModel::columnCount(const QModelIndex&) const
{
	if (not _nraw) {
		return 0;
	}
	return _nraw->column_count();
}

QVariant PVInspector::PVNrawListingModel::data(const QModelIndex& index, int role) const
{
	if (not _nraw) {
		return {};
	}

	bool is_element_valid = _nraw->valid_rows_sel().get_line_fast(index.row());

	switch (role) {
	case Qt::DisplayRole:
		if (is_element_valid) {
			return QString::fromStdString(_nraw->at_string(index.row(), (PVCol)index.column()));
		} else {
			auto it = _inv_elts.find(index.row());
			if (it != _inv_elts.end()) {
				return QString::fromStdString(_inv_elts.at(index.row()));
			}
		}
		break;
	case Qt::BackgroundRole: {
		if (_show_sel && index.column() == _col_tosel) {
			// TODO: put this color in something more global (taken from PVListingModel.cpp)
			return QColor(88, 172, 250);
		}
		if (not is_element_valid) {
			return QColor(250, 197, 205);
		}
		break;
	}
	case (Qt::FontRole): {
		QFont f;
		f.setItalic(not is_element_valid or
		            not _nraw->column((PVCol)index.column()).is_valid(index.row()));
		return f;
	}
	};

	return {};
}

Qt::ItemFlags PVInspector::PVNrawListingModel::flags(const QModelIndex& /*index*/) const
{
	return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}

QVariant PVInspector::PVNrawListingModel::headerData(int section,
                                                     Qt::Orientation orientation,
                                                     int role) const
{
	if (orientation == Qt::Vertical && role == Qt::DisplayRole) {
		return QAbstractTableModel::headerData(section + _starting_row, orientation, role);
	}
	if (orientation == Qt::Horizontal && role == Qt::DisplayRole) {
		return _format.get_axes().at(section).get_name();
	}

	return QAbstractTableModel::headerData(section, orientation, role);
}

void PVInspector::PVNrawListingModel::set_nraw(PVRush::PVNraw const& nraw)
{
	if (nraw.row_count() == 0) {
		_nraw = nullptr;
	} else {
		_nraw = &nraw;
	}
	Q_EMIT layoutChanged();
}

void PVInspector::PVNrawListingModel::set_selected_column(PVCol col)
{
	_col_tosel = col;
}

void PVInspector::PVNrawListingModel::sel_visible(bool visible)
{
	_show_sel = visible;
	Q_EMIT layoutChanged();
}
