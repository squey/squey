/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2015
 */

#ifndef __PVGUIQT_PVABSTRACTSTATSMODEL_H__
#define __PVGUIQT_PVABSTRACTSTATSMODEL_H__

#include <QAbstractListModel>

namespace PVGuiQt {

class PVAbstractStatsModel: public QAbstractListModel
{

public:
	PVAbstractStatsModel(QWidget* parent = nullptr) : QAbstractListModel(parent) {}

public:
	QVariant headerData(int section, Qt::Orientation orientation, int role) const
	{
		static QString h[] = { "Value", "Count " };

		switch (role) {
			case(Qt::DisplayRole) :
			{
				if (orientation == Qt::Horizontal) {
					if (section == 1) {
						return count_header();
					}
					return h[section];
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

	int columnCount(const QModelIndex& /*index*/) const
	{
		return 2;
	}

public:
	void use_logarithmic_scale(bool log_scale)
	{
		_use_logarithmic_scale = log_scale;
	}

	void use_absolute_max_count(bool abs_max)
	{
		_use_absolute_max_count = abs_max;
	}

	static inline QString format_occurence(double occurence_count)
	{
		double intpart;
		bool integer = std::modf(occurence_count, &intpart) == (double) 0;

		return integer ? QString("%L1").arg((int64_t) occurence_count) : QString("%L1").arg(occurence_count, 0, 'g');

	};
	static inline QString format_percentage(double ratio) { return QLocale().toString(ratio * 100, 'f', 1) + "%"; };
	static inline QString format_scientific_notation(double ratio) { return QLocale().toString(ratio, 'e', 1); };

private:
	QString count_header() const
	{
		return QString("Count ") + " (" + ( _use_logarithmic_scale ? "Log" : "Lin") + "/" + (_use_absolute_max_count ? "Abs" : "Rel") + ")";
	}

private:
	bool _use_logarithmic_scale;
	bool _use_absolute_max_count;

};

} // namespace PVGuiQt

#endif // __PVGUIQT_PVABSTRACTSTATSMODEL_H__
