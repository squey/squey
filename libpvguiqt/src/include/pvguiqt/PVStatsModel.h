/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2015
 */

#ifndef __PVGUIQT_PVSTATSMODEL_H__
#define __PVGUIQT_PVSTATSMODEL_H__

#include <pvguiqt/PVAbstractTableModel.h>

#include <pvkernel/core/inendi_bench.h>

#include <pvcop/db/types.h>

#include <boost/date_time/posix_time/posix_time.hpp>

namespace PVGuiQt
{

enum ValueFormat { Count = 1, Percent = 2, Scientific = 4 };

class PVStatsModel : public PVAbstractTableModel
{
  public:
	PVStatsModel(const QString& op_name,
	             const QString& col1_name,
	             const QString& col2_name,
	             pvcop::db::array col1,
	             pvcop::db::array col2,
	             pvcop::db::array abs_max,
	             pvcop::db::array minmax,
	             QWidget* parent = nullptr)
	    : PVAbstractTableModel(col1.size(), parent)
	    , _op_name(op_name)
	    , _col1_name(col1_name)
	    , _col2_name(col2_name)
	    , _col1(std::move(col1))
	    , _col2(std::move(col2))
	    , _format(ValueFormat::Count)
	    , _abs_max(std::move(abs_max))
	    , _minmax(std::move(minmax))
	{
	}

	QVariant headerData(int section, Qt::Orientation orientation, int role) const override
	{
		switch (role) {
		case (Qt::DisplayRole): {
			if (orientation == Qt::Horizontal) {
				if (section == 1) {
					return count_header();
				}
				return _col1_name;
			}

			return row_pos(section) + 1; // Start counting rows from 1 for display
		} break;
		case (Qt::TextAlignmentRole):
			if (orientation == Qt::Horizontal) {
				return (Qt::AlignLeft + Qt::AlignVCenter);
			} else {
				return (Qt::AlignRight + Qt::AlignVCenter);
			}
			break;
		case Qt::InitialSortOrderRole:
			// this role is to get the default sort order of a section
			if (orientation == Qt::Vertical) {
				// handling vertical header is not relevant
				return QVariant();
			}
			if (section == 0) {
				// the values column uses ascending
				return Qt::AscendingOrder;
			} else {
				// the stats column uses descending
				return Qt::DescendingOrder;
			}
			break;
		default:
			return QVariant();
			break;
		}

		return QVariant();
	}

	QString export_line(int row, const QString& fsep) const override
	{
		static const QString escaped_quote("\"\"");
		static const QString quote("\"");

		QString value = QString::fromStdString(_col1.at(row));

		if (!_copy_count) {
			return value;
		}

		// Escape quotes
		value.replace(quote, escaped_quote);
		value = quote + value + quote;

		double occurence_count = QString::fromStdString(_col2.at(row)).toDouble();

		double ratio = occurence_count / max_count();
		if ((_format & ValueFormat::Count) == ValueFormat::Count) {
			value.append(fsep + quote + format_occurence(occurence_count) + quote);
		}
		if ((_format & ValueFormat::Scientific) == ValueFormat::Scientific) {
			value.append(fsep + quote + format_scientific_notation(ratio) + quote);
		}
		if ((_format & ValueFormat::Percent) == ValueFormat::Percent) {
			value.append(fsep + quote + format_percentage(ratio) + quote);
		}

		return value;
	}

  public:
	QVariant data(QModelIndex const& index, int role) const override
	{
		if (not index.isValid()) {
			return {};
		}

		size_t row = rowIndex(index);
		assert(row < _col1.size());

		switch (role) {
		case Qt::DisplayRole:
			if (index.column() == 0) {
				std::string const& str = _col1.at(row);
				return QString::fromUtf8(str.c_str(), str.size());
			}
			break;
		case Qt::ToolTipRole: {
			std::string const& raw_str = _col1.at(row);
			return get_wrapped_string(QString::fromUtf8(raw_str.c_str(), raw_str.size()));
		}

		case Qt::UserRole:
			if (index.column() == 1) {
				std::string const& str = _col2.at(row);
				return QString::fromUtf8(str.c_str(), str.size());
			}
			break;
		case Qt::BackgroundRole:
			if (is_selected(index)) {
				return _selection_brush;
			}
			break;
		case (Qt::FontRole): {
			if (index.column() == 0) {
				QFont f;
				if (not _col1.is_valid(row)) {
					f.setItalic(true); // Set invalid values in italic
				}
				return f;
			}
		}
		default:
			break;
		}

		return QVariant();
	}

	void sort(int col_idx, Qt::SortOrder order) override
	{
		assert(col_idx == 0 || col_idx == 1);

		Q_EMIT layoutAboutToBeChanged();

		if (_display.sorted_column() != col_idx) {
			const pvcop::db::array& column = (col_idx == 0) ? _col1 : _col2;

			BENCH_START(sort);
			_display.sorting() = column.parallel_sort();
			BENCH_END(sort, "sort", column.size(), /*column.mem_size() / column.size()*/ 1,
			          column.size(), /*column.mem_size() / column.size()*/ 1);
		}

		// FIXME(pbrunet) : What if we cancel it?
		sorted(PVCombCol(col_idx), order);
		_display.set_filter_as_sort();

		Q_EMIT layoutChanged();
	}

	int columnCount(const QModelIndex& /*index*/) const override { return 2; }

	void use_logarithmic_scale(bool log_scale) { _use_logarithmic_scale = log_scale; }

	static inline QString format_occurence(double occurence_count)
	{
		double intpart;
		bool integer = std::modf(occurence_count, &intpart) == (double)0;

		return integer ? QString("%L1").arg((int64_t)occurence_count)
		               : QString("%L1").arg(occurence_count, 0, 'f', 3);
	};
	static inline QString format_percentage(double ratio)
	{
		return QLocale().toString(ratio * 100, 'f', 1) + "%";
	};
	static inline QString format_scientific_notation(double ratio)
	{
		return QLocale().toString(ratio, 'e', 1);
	};

	void set_copy_count(bool v) { _copy_count = v; }
	void set_use_absolute(bool a) { _use_absolute_max_count = a; }
	void set_format(ValueFormat f, bool e)
	{
		if (e) {
			_format = (ValueFormat)(_format | f);
		} else {
			_format = (ValueFormat)(_format & ~f);
		}
	}
	void set_use_log_scale(bool log_scale) { _use_logarithmic_scale = log_scale; }

  private:
	QString count_header() const
	{
		return _col2_name.isNull() ? "Count" : _op_name + " on " + _col2_name;
	}

  public:
	inline double max_count() const
	{
		return _use_absolute_max_count ? absolute_max_count() : relative_max_count();
	}
	inline double relative_max_count() const { return as_double(_minmax, 1); }
	inline double relative_min_count() const { return as_double(_minmax, 0); }
	inline double absolute_max_count() const { return as_double(_abs_max, 0); }
	bool use_log_scale() const { return _use_logarithmic_scale; }
	pvcop::db::array const& value_col() const { return _col1; }
	pvcop::db::array const& stat_col() const { return _col2; }

	double stat_as_double(size_t row) const { return as_double(_col2, row); }

  private:
	double as_double(const pvcop::db::array& array, size_t row) const
	{
		if (array.type() == "duration") {
			const auto& duration_array = array.to_core_array<boost::posix_time::time_duration>();
			return (double)duration_array[row].total_microseconds();
		} else {
			return QString::fromStdString(array.at(row)).toDouble();
		}
	}

  private:
	using type_index = pvcop::db::index_t;

  private:
	QString _op_name;             // <! Name of the operation
	QString _col1_name;           // <! Name of the first column
	QString _col2_name;           // <! Name of the second column
	const pvcop::db::array _col1; //!< Values handled.
	const pvcop::db::array _col2; //!< Number of occurance for each values.

	ValueFormat _format; //!< Export format (percent, scientific or count)
	// This variable is always set right before the copy. We use attribut to keep
	// Export generic with other TableModels.
	bool _copy_count; //!< If on line copy, we also copy values.

	const pvcop::db::array _abs_max;
	const pvcop::db::array _minmax;

	bool _use_absolute_max_count =
	    true; // FIXME : should not be in the model as it only concern the display
	bool _use_logarithmic_scale =
	    true; // FIXME : should not be in the model as it only concern the display
};

} // namespace PVGuiQt

#endif // __PVGUIQT_PVSTATSMODEL_H__
