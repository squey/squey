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
#include <QDoubleSpinBox>
#include <QStyle>
#include <QLineEdit>
#include <QWidget>
#include <QSpinBox>

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/widgets/PVLongLongSpinBox.h>
#include <pvcop/db/array.h>
#include <pvcop/formatter_desc.h>

namespace PVWidgets
{

class PVRangeEdit : public QWidget
{
  public:
	using func_type = std::function<void(const pvcop::db::array& minmax)>;

  public:
	PVRangeEdit(const pvcop::db::array& minmax, func_type f, QWidget* parent = nullptr)
	    : QWidget(parent), _minmax(minmax.copy()), _validate_f(f)
	{
	}
	virtual ~PVRangeEdit() {}

  public:
	const pvcop::db::array& minmax() const { return _minmax; }
	virtual void set_minmax(const pvcop::db::array& minmax) = 0;

  protected:
	pvcop::db::array _minmax;
	func_type _validate_f;
};

template <typename SpinType, typename T>
class PVNumberRangeEdit : public PVRangeEdit
{
  public:
	PVNumberRangeEdit(const pvcop::db::array& minmax, func_type f, QWidget* parent = nullptr)
	    : PVRangeEdit(minmax, f, parent)
	{
		_from_widget = new SpinType;
		_to_widget = new SpinType;
		set_minmax(minmax);
		_ok = new QPushButton("&Ok");

		QHBoxLayout* layout = new QHBoxLayout;
		layout->addWidget(_from_widget);
		layout->addWidget(_to_widget);
		layout->addWidget(_ok);
		layout->addStretch();

		setLayout(layout);

		connect(_from_widget, static_cast<void (SpinType::*)(T)>(&SpinType::valueChanged),
		        [&](T value) {
			        _to_widget->setMinimum(value);
			        update_minmax(value, false);
		        });
		connect(_to_widget, static_cast<void (SpinType::*)(T)>(&SpinType::valueChanged),
		        [&](T value) {
			        _from_widget->setMaximum(value);
			        update_minmax(value, true);
		        });
		connect(_ok, &QPushButton::clicked, [&]() { _validate_f(_minmax); });
	}

	void set_minmax(const pvcop::db::array& minmax) override
	{
		_minmax = minmax.copy();

		assert(_from_widget);
		assert(_to_widget);

		qlonglong min = QString(_minmax.at(0).c_str()).toLongLong();
		qlonglong max = QString(_minmax.at(1).c_str()).toLongLong();

		int spin_width = QFontMetrics(_from_widget->font()).width(QString::number(max)) + 25;
		_from_widget->setFixedWidth(spin_width);
		_from_widget->setMinimum(min);
		_from_widget->setMaximum(max);
		_from_widget->setValue(min);
		_to_widget->setFixedWidth(spin_width);
		_to_widget->setMinimum(min);
		_to_widget->setMaximum(max);
		_to_widget->setValue(max);
	}

  private:
	void update_minmax(qlonglong value, bool max)
	{
		const pvcop::types::formatter_interface::shared_ptr& dtf = _minmax.formatter();
		dtf->from_string(std::to_string(value).c_str(), _minmax.data(), max);
	}

  private:
	SpinType* _from_widget;
	SpinType* _to_widget;
	QPushButton* _ok;
};

class PVLongLongRangeEdit : public PVNumberRangeEdit<PVWidgets::PVLongLongSpinBox, qlonglong>
{
  public:
	PVLongLongRangeEdit(const pvcop::db::array& minmax, func_type f, QWidget* parent = nullptr)
	    : PVNumberRangeEdit<PVWidgets::PVLongLongSpinBox, qlonglong>(minmax, f, parent)
	{
	}
};

class PVIntegerRangeEdit : public PVNumberRangeEdit<QSpinBox, int>
{
  public:
	PVIntegerRangeEdit(const pvcop::db::array& minmax, func_type f, QWidget* parent = nullptr)
	    : PVNumberRangeEdit<QSpinBox, int>(minmax, f, parent)
	{
	}
};

class PVDoubleRangeEdit : public PVRangeEdit
{
  public:
	PVDoubleRangeEdit(const pvcop::db::array& minmax, func_type f, QWidget* parent = nullptr)
	    : PVRangeEdit(minmax, f, parent)
	{
		_from_widget = new QDoubleSpinBox;
		_to_widget = new QDoubleSpinBox;
		set_minmax(minmax);
		_ok = new QPushButton("&Ok");

		QHBoxLayout* layout = new QHBoxLayout;
		layout->addWidget(_from_widget);
		layout->addWidget(_to_widget);
		layout->addWidget(_ok);
		layout->addStretch();

		setLayout(layout);

		connect(_from_widget,
		        static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged),
		        [&](double value) {
			        _to_widget->setMinimum(value);
			        update_minmax(value, false);
		        });
		connect(_to_widget,
		        static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged),
		        [&](double value) {
			        _from_widget->setMaximum(value);
			        update_minmax(value, true);
		        });
		connect(_ok, &QPushButton::clicked, [&]() { _validate_f(_minmax); });
	}

	void set_minmax(const pvcop::db::array& minmax) override
	{
		_minmax = minmax.copy();

		assert(_from_widget);
		assert(_to_widget);

		double min = QString(_minmax.at(0).c_str()).toDouble();
		double max = QString(_minmax.at(1).c_str()).toDouble();

		// int spin_width = QFontMetrics(_from_widget->font()).width(QString::number(max)) + 25;
		//_from_widget->setFixedWidth(spin_width);
		_from_widget->setMinimum(min);
		_from_widget->setMaximum(max);
		_from_widget->setValue(min);
		//_to_widget->setFixedWidth(spin_width);
		_to_widget->setMinimum(min);
		_to_widget->setMaximum(max);
		_to_widget->setValue(max);
	}

  private:
	void update_minmax(double value, bool max)
	{
		const pvcop::types::formatter_interface::shared_ptr& dtf = _minmax.formatter();
		dtf->from_string(std::to_string(value).c_str(), _minmax.data(), max);
	}

  private:
	QDoubleSpinBox* _from_widget;
	QDoubleSpinBox* _to_widget;
	QPushButton* _ok;
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
	PVDateTimeRangeEdit(const pvcop::db::array& minmax, func_type f, QWidget* parent = nullptr)
	    : PVRangeEdit(minmax, f, parent)
	{
		_from_widget = new QDateTimeEdit;
		_to_widget = new QDateTimeEdit;
		set_minmax(minmax);

		_ok = new QPushButton("&Ok");

		QHBoxLayout* layout = new QHBoxLayout;
		layout->addWidget(_from_widget);
		layout->addWidget(_to_widget);
		layout->addWidget(_ok);
		layout->addStretch();

		setLayout(layout);

		connect(_from_widget, &QDateTimeEdit::dateTimeChanged, [&](const QDateTime& datetime) {
			_to_widget->setMinimumDateTime(datetime);
			update_minmax(datetime, false);
		});
		connect(_to_widget, &QDateTimeEdit::dateTimeChanged, [&](const QDateTime& datetime) {
			_from_widget->setMaximumDateTime(datetime);
			update_minmax(datetime, true);
		});
		connect(_ok, &QPushButton::clicked, [&]() { _validate_f(_minmax); });
	}

	void set_minmax(const pvcop::db::array& minmax) override
	{
		_minmax = minmax.copy();

		assert(_from_widget);
		assert(_to_widget);

		const char* parse_format_inendi;
		const char* parse_format_qt;
		const char* display_format_qt;
		size_t trim_size;
		if (_minmax.formatter()->name() == "datetime") {
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
		_minmax.formatter()->set_parameters(formatter_desc.parameters().c_str());

		const std::string& min_date_str = _minmax.at(0);
		const std::string& max_date_str = _minmax.at(1);
		const QDateTime& min_date = QDateTime::fromString(
		    QString::fromStdString(min_date_str).left(min_date_str.size() - trim_size),
		    parse_format_qt);
		const QDateTime& max_date = QDateTime::fromString(
		    QString::fromStdString(max_date_str).left(max_date_str.size() - trim_size),
		    parse_format_qt);

		_from_widget->setMinimumDateTime(min_date);
		_from_widget->setMaximumDateTime(max_date);
		_from_widget->setDateTime(min_date);
		_to_widget->setMinimumDateTime(min_date);
		_to_widget->setMaximumDateTime(max_date);
		_to_widget->setDateTime(max_date);

		_from_widget->setCalendarPopup(true);
		_to_widget->setCalendarPopup(true);

		_from_widget->setDisplayFormat(display_format_qt);
		_to_widget->setDisplayFormat(display_format_qt);
	}

  private:
	void update_minmax(const QDateTime& datetime, bool max)
	{
		const pvcop::types::formatter_interface::shared_ptr& dtf = _minmax.formatter();
		dtf->from_string(datetime.toString(parse_format_us_qt).toStdString().c_str(),
		                 _minmax.data(), max);
	}

  private:
	QDateTimeEdit* _from_widget;
	QDateTimeEdit* _to_widget;
	QPushButton* _ok;
};

class PVDurationRangeEdit : public PVRangeEdit
{
	using func_type = std::function<void(const pvcop::db::array& minmax)>;

  public:
	PVDurationRangeEdit(const pvcop::db::array& minmax, func_type f, QWidget* parent = nullptr)
	    : PVRangeEdit(minmax, f, parent)
	{
		_from_widget = new QLineEdit;
		_to_widget = new QLineEdit;
		_ok = new QPushButton("&Ok");

		// Setup duration validator
		QRegularExpression duration_re("\\b[0-9]+:[0-5][0-9]:[0-5][0-9](\\.[0-9]{6})?\\b",
		                               QRegularExpression::CaseInsensitiveOption);
		QRegularExpressionValidator* re_val = new QRegularExpressionValidator(duration_re, this);
		_from_widget->setValidator(re_val);
		_to_widget->setValidator(re_val);
		auto check_durations_valid_f = [&, re_val]() {
			QString from = _from_widget->text();
			QString to = _to_widget->text();
			int pos = 0;
			bool from_valid = re_val->validate(from, pos) == QValidator::Acceptable;
			bool to_valid = re_val->validate(to, pos) == QValidator::Acceptable;
			const QString& style_error = "QLineEdit{border: 1px solid red;}";
			if (not from_valid) {
				_from_widget->setStyleSheet(style_error);
			} else {
				_from_widget->setStyleSheet("");
			}
			if (not to_valid) {
				_to_widget->setStyleSheet(style_error);
			} else {
				_to_widget->setStyleSheet("");
			}
			_ok->setEnabled(from_valid and to_valid);
		};
		connect(_from_widget, &QLineEdit::textChanged, check_durations_valid_f);
		connect(_to_widget, &QLineEdit::textChanged, check_durations_valid_f);

		set_minmax(minmax);

		QHBoxLayout* layout = new QHBoxLayout;
		layout->addWidget(_from_widget);
		layout->addWidget(_to_widget);
		layout->addWidget(_ok);
		layout->addStretch();

		setLayout(layout);

		connect(_from_widget, &QLineEdit::textChanged,
		        [this]() { update_minmax(_from_widget->text(), false); });
		connect(_to_widget, &QLineEdit::textChanged,
		        [this]() { update_minmax(_to_widget->text(), true); });
		connect(_ok, &QPushButton::clicked, [this]() { _validate_f(_minmax); });
	}

	void set_minmax(const pvcop::db::array& minmax) override
	{
		_minmax = minmax.copy();

		assert(_from_widget);
		assert(_to_widget);

		const std::string& min_duration_str = _minmax.at(0);
		const std::string& max_duration_str = _minmax.at(1);

		_from_widget->setText(min_duration_str.c_str());
		_to_widget->setText(max_duration_str.c_str());
	}

  private:
	void update_minmax(const QString& duration, bool max)
	{
		const pvcop::types::formatter_interface::shared_ptr& dtf = _minmax.formatter();
		dtf->from_string(duration.toStdString().c_str(), _minmax.data(), max);
	}

  private:
	QLineEdit* _from_widget;
	QLineEdit* _to_widget;
	QPushButton* _ok;
};

class PVRangeEditFactory
{
  public:
	static PVRangeEdit* create(const pvcop::db::array& minmax, PVRangeEdit::func_type f)
	{
		PVWidgets::PVRangeEdit* range_edit = nullptr;

		if (minmax.formatter()->name().find("datetime") == 0) {
			range_edit = new PVWidgets::PVDateTimeRangeEdit(minmax, f);
		} else if (minmax.formatter()->name().find("duration") == 0) {
			range_edit = new PVWidgets::PVDurationRangeEdit(minmax, f);
		} else if (minmax.formatter()->name().find("number_float") == 0 or
		           minmax.formatter()->name().find("number_double") == 0) {
			range_edit = new PVWidgets::PVDoubleRangeEdit(minmax, f);
		} else if (minmax.formatter()->name().find("number_uint") == 0) {
			range_edit = new PVWidgets::PVLongLongRangeEdit(minmax, f);
		} else if (minmax.formatter()->name().find("number_int") == 0) {
			range_edit = new PVWidgets::PVIntegerRangeEdit(minmax, f);
		}

		return range_edit;
	}
};

} // namespace PVWidgets

#endif // __PVWIDGETS_RANGEEDIT_H__
