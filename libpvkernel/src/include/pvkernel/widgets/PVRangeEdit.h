/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2018
 */

#ifndef __PVWIDGETS_RANGEEDIT_H__
#define __PVWIDGETS_RANGEEDIT_H__

#include <QApplication>
#include <QDateTimeEdit>
#include <QHBoxLayout>
#include <QPushButton>
#include <QSpinBox>
#include <QStyle>
#include <QWidget>

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/widgets/PVLongLongSpinBox.h>
#include <pvcop/db/array.h>
#include <pvcop/formatter_desc.h>

namespace PVWidgets
{

class PVRangeEdit : public QWidget
{
  public:
	PVRangeEdit(QWidget* parent = nullptr) : QWidget(parent) {}
	virtual ~PVRangeEdit() {}
};

class PVIntegerRangeEdit : public PVRangeEdit
{
  private:
	using func_type = std::function<void(const pvcop::db::array& minmax)>;

  public:
	PVIntegerRangeEdit(const pvcop::db::array& minmax, func_type f, QWidget* parent = nullptr)
	    : PVRangeEdit(parent), _minmax(minmax.copy()), _validate_f(f)
	{
		qlonglong min = QString(minmax.at(0).c_str()).toLongLong();
		qlonglong max = QString(minmax.at(1).c_str()).toLongLong();

		_from_widget = new PVWidgets::PVLongLongSpinBox;
		int spin_width = QFontMetrics(_from_widget->font()).width(QString::number(max)) + 25;
		_from_widget->setFixedWidth(spin_width);
		_from_widget->setMinimum(min);
		_from_widget->setMaximum(max);
		_from_widget->setValue(min);
		_to_widget = new PVWidgets::PVLongLongSpinBox;
		_to_widget->setFixedWidth(spin_width);
		_to_widget->setMinimum(min);
		_to_widget->setMaximum(max);
		_to_widget->setValue(max);
		_ok = new QPushButton("&Ok");

		QHBoxLayout* layout = new QHBoxLayout;
		layout->addWidget(_from_widget);
		layout->addWidget(_to_widget);
		layout->addWidget(_ok);
		layout->addStretch();

		setLayout(layout);

		connect(_from_widget, static_cast<void (PVWidgets::PVLongLongSpinBox::*)(qlonglong)>(
		                          &PVWidgets::PVLongLongSpinBox::valueChanged),
		        [&](qlonglong value) {
			        _to_widget->setMinimum(value);
			        update_minmax(value, false);
			    });
		connect(_to_widget, static_cast<void (PVWidgets::PVLongLongSpinBox::*)(qlonglong)>(
		                        &PVWidgets::PVLongLongSpinBox::valueChanged),
		        [&](qlonglong value) {
			        _from_widget->setMaximum(value);
			        update_minmax(value, true);
			    });
		connect(_ok, &QPushButton::clicked, [&]() { _validate_f(_minmax); });
	}

  public:
	const pvcop::db::array& minmax() const { return _minmax; }

  private:
	void update_minmax(qlonglong value, bool max)
	{
		const pvcop::types::formatter_interface::shared_ptr& dtf = _minmax.formatter();
		dtf->from_string(std::to_string(value).c_str(), _minmax.data(), max);
	}

  private:
	PVWidgets::PVLongLongSpinBox* _from_widget;
	PVWidgets::PVLongLongSpinBox* _to_widget;
	QPushButton* _ok;

	pvcop::db::array _minmax;
	func_type _validate_f;
};

class PVDateTimeRangeEdit : public PVRangeEdit
{
  private:
	static constexpr const char parse_format_us_inendi[] = "yyyy-MM-dd HH:mm:ss.S";
	static constexpr const char parse_format_sec_inendi[] = "yyyy-MM-dd HH:mm:ss";
	static constexpr const char parse_format_us_qt[] = "yyyy-MM-dd HH:mm:ss.zzz";
	static constexpr const char parse_format_sec_qt[] = "yyyy-MM-dd HH:mm:ss";
	static constexpr const char display_format_us_qt[] = "yyyy-MMM-dd HH:mm:ss.z";
	static constexpr const char display_format_sec_qt[] = "yyyy-MMM-dd HH:mm:ss";

	using func_type = std::function<void(const pvcop::db::array& minmax)>;

  public:
	PVDateTimeRangeEdit(const pvcop::db::array& minmax_time, func_type f, QWidget* parent = nullptr)
	    : PVRangeEdit(parent), _minmax_time(minmax_time.copy()), _validate_f(f)
	{

		const char* parse_format_inendi;
		const char* parse_format_qt;
		const char* display_format_qt;
		size_t trim_size;
		if (minmax_time.formatter()->name() == "datetime") {
			parse_format_inendi = parse_format_sec_inendi;
			parse_format_qt = parse_format_sec_qt;
			display_format_qt = display_format_sec_qt;
			trim_size = 0;
		} else {
			parse_format_inendi = parse_format_us_inendi;
			parse_format_qt = parse_format_us_qt;
			display_format_qt = display_format_us_qt;
			trim_size = 3;
		}

		const pvcop::formatter_desc& formatter_desc =
		    PVRush::PVFormat::get_datetime_formatter_desc(std::string(parse_format_inendi));
		_minmax_time.formatter()->set_parameters(formatter_desc.parameters().c_str());

		const std::string& min_date_str = _minmax_time.at(0);
		const std::string& max_date_str = _minmax_time.at(1);
		const QDateTime& min_date = QDateTime::fromString(
		    QString::fromStdString(min_date_str).left(min_date_str.size() - trim_size),
		    parse_format_qt);
		const QDateTime& max_date = QDateTime::fromString(
		    QString::fromStdString(max_date_str).left(max_date_str.size() - trim_size),
		    parse_format_qt);

		_from_datetime_widget = new QDateTimeEdit(min_date);
		_from_datetime_widget->setMinimumDateTime(min_date);
		_from_datetime_widget->setMaximumDateTime(max_date);
		_to_datetime_widget = new QDateTimeEdit(max_date);
		_to_datetime_widget->setMinimumDateTime(min_date);
		_to_datetime_widget->setMaximumDateTime(max_date);

		_from_datetime_widget->setCalendarPopup(true);
		_to_datetime_widget->setCalendarPopup(true);

		_from_datetime_widget->setDisplayFormat(display_format_qt);
		_to_datetime_widget->setDisplayFormat(display_format_qt);

		_ok = new QPushButton("&Ok");

		QHBoxLayout* layout = new QHBoxLayout;
		layout->addWidget(_from_datetime_widget);
		layout->addWidget(_to_datetime_widget);
		layout->addWidget(_ok);
		layout->addStretch();

		setLayout(layout);

		connect(_from_datetime_widget, &QDateTimeEdit::dateTimeChanged,
		        [&](const QDateTime& datetime) {
			        _to_datetime_widget->setMinimumDateTime(datetime);
			        update_minmax_time(datetime, false);
			    });
		connect(_to_datetime_widget, &QDateTimeEdit::dateTimeChanged,
		        [&](const QDateTime& datetime) {
			        _from_datetime_widget->setMaximumDateTime(datetime);
			        update_minmax_time(datetime, true);
			    });
		connect(_ok, &QPushButton::clicked, [&]() { _validate_f(_minmax_time); });
	}

  public:
	const pvcop::db::array& minmax_time() const { return _minmax_time; }

  private:
	void update_minmax_time(const QDateTime& datetime, bool max)
	{
		const pvcop::types::formatter_interface::shared_ptr& dtf = _minmax_time.formatter();
		dtf->from_string(datetime.toString(parse_format_us_qt).toStdString().c_str(),
		                 _minmax_time.data(), max);
	}

  private:
	QDateTimeEdit* _from_datetime_widget;
	QDateTimeEdit* _to_datetime_widget;
	QPushButton* _ok;

	pvcop::db::array _minmax_time;
	func_type _validate_f;
};

} // namespace PVWidgets

#endif // __PVWIDGETS_RANGEEDIT_H__
