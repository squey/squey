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

namespace PVGuiQt
{

enum ValueFormat
{
	Count		= 0,
	Percent		= 1,
	Scientific	= 2
};

class PVStatsModel: public PVAbstractTableModel
{
public:
	PVStatsModel(pvcop::db::array col1, pvcop::db::array col2, double absolute_max, double relative_min, double relative_max, QWidget* parent = nullptr)
		: PVAbstractTableModel(col1.size(), parent),
		 _col1(std::move(col1)),
		 _col2(std::move(col2)),
		 _absolute_max_count(absolute_max),
		 _relative_min_count(relative_min),
		 _relative_max_count(relative_max)
	{
	}

	QVariant headerData(int section, Qt::Orientation orientation, int role) const
	{
		switch (role) {
			case(Qt::DisplayRole) :
			{
				if (orientation == Qt::Horizontal) {
					if (section == 1) {
						return count_header();
					}
					return "Value";
				}
				return QVariant(QString().setNum(section));
			}
			break;
			case (Qt::TextAlignmentRole) :
				if (orientation == Qt::Horizontal) {
					return (Qt::AlignLeft + Qt::AlignVCenter);
				}
				else {
					return (Qt::AlignRight + Qt::AlignVCenter);
				}
			break;
			default:
				return QVariant();
			break;
		}

		return QVariant();
	}

	QString export_line(int row) const override
	{
		static const QString sep(",");
		static const QString escaped_quote("\"\"");
		static const QString quote("\"");

		QString value = QString::fromStdString(_col1.at(row));

		if(!_copy_count) {
			return value;
		}

		// Escape quotes
		value.replace(quote, escaped_quote);
		value = quote + value + quote + sep;

		double occurence_count = QString::fromStdString(_col2.at(row_pos_to_index(row))).toDouble();

		double ratio = occurence_count / max_count();
		if ((_format & ValueFormat::Count) == ValueFormat::Count) {
			value.append(quote + format_occurence(occurence_count) + quote + sep);
		}
		if ((_format & ValueFormat::Scientific) == ValueFormat::Scientific) {
			value.append(quote + format_scientific_notation(ratio) + quote + sep);
		}
		if ((_format & ValueFormat::Percent) == ValueFormat::Percent) {
			value.append(quote + format_percentage(ratio) + quote + sep);
		}
		return value;
	}

public:
	QVariant data(QModelIndex const& index, int role) const
	{
		size_t row = rowIndex(index);
		assert(row < _col1.size());

		switch(role) {
			case Qt::DisplayRole:
				if (index.column() == 0) {
					std::string const& str = _col1.at(row);
					return QString::fromUtf8(str.c_str(), str.size());
				}
				break;
			case Qt::UserRole:
				if(index.column() == 1) {
					std::string const& str = _col2.at(row);
					return QString::fromUtf8(str.c_str(), str.size());
				}
				break;
			case Qt::BackgroundRole:
				if (is_selected(index)) {
					return _selection_brush;
				}
				break;
			default:
				break;
		}

		return QVariant();
	}

	void sort(int col_idx, Qt::SortOrder order) override
	{
		assert(col_idx == 0 || col_idx == 1);

		if(sorted_column() != col_idx) {
			const pvcop::db::array& column = (col_idx == 0) ? _col1 : _col2;

			BENCH_START(sort);
			sorting().parallel_sort_on(column);
			BENCH_END(sort, "sort", column.size(), /*column.mem_size() / column.size()*/1, column.size(), /*column.mem_size() / column.size()*/1);
		}

		// FIXME(pbrunet) : What ifwe cancel it?
		sorted(col_idx, order);
		filter_is_sort();

		emit layoutChanged();
	}

	int columnCount(const QModelIndex& /*index*/) const
	{
		return 2;
	}

	void use_logarithmic_scale(bool log_scale)
	{
		_use_logarithmic_scale = log_scale;
	}

	static inline QString format_occurence(double occurence_count)
	{
		double intpart;
		bool integer = std::modf(occurence_count, &intpart) == (double) 0;

		return integer ? QString("%L1").arg((int64_t) occurence_count) : QString("%L1").arg(occurence_count, 0, 'g');

	};
	static inline QString format_percentage(double ratio) { return QLocale().toString(ratio * 100, 'f', 1) + "%"; };
	static inline QString format_scientific_notation(double ratio) { return QLocale().toString(ratio, 'e', 1); };

	void set_copy_count(bool v) { _copy_count = v; }
	void set_use_absolute(bool a) { _use_absolute_max_count = a; }
	void set_format(ValueFormat f, bool e) {if (e) { _format = (ValueFormat) (_format | f); } else { _format = (ValueFormat) (_format & ~f); } }
	void set_use_log_scale(bool log_scale) { _use_logarithmic_scale = log_scale; }

private:
	QString count_header() const
	{
		return QString("Count ") + " (" + ( _use_logarithmic_scale ? "Log" : "Lin") + "/" + (_use_absolute_max_count ? "Abs" : "Rel") + ")";
	}

public:
	inline double max_count() const { return _use_absolute_max_count ? _absolute_max_count : _relative_max_count; }
	inline double relative_max_count() const { return _relative_max_count; }
	inline double relative_min_count() const { return _relative_min_count; }
	inline double absolute_max_count() const { return _absolute_max_count; }
	bool use_log_scale() const { return _use_logarithmic_scale; }
	pvcop::db::array const& value_col() const { return _col1; }
	pvcop::db::array const& stat_col() const { return _col2; }

private:
	using type_index = typename pvcop::db::type_traits::type_id<pvcop::db::type_index>::type;

private:
	const pvcop::db::array _col1; //!< Values handled.
	const pvcop::db::array _col2; //!< Number of occurance for each values.

	ValueFormat _format; //!< Export format (percent, scientific or count)
	// This variable is always set right before the copy. We use attribut to keep
	// Export generic with other TableModels.
	bool _copy_count; //!< If on line copy, we also copy values.

	double _absolute_max_count;
	double _relative_min_count;
	double _relative_max_count;

	bool _use_absolute_max_count = true; // FIXME : should not be in the model as it only concern the display
	bool _use_logarithmic_scale = true; // FIXME : should not be in the model as it only concern the display
};

} // namespace PVGuiQt

#endif // __PVGUIQT_PVSTATSMODEL_H__
