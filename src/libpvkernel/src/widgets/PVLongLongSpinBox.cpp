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

#include <pvkernel/widgets/PVLongLongSpinBox.h>
#include <qminmax.h>
#include <qnamespace.h>
#include <QLineEdit>
#include <limits>
#include <QKeyEvent>

class QWidget;

PVWidgets::PVLongLongSpinBox::PVLongLongSpinBox(QWidget* parent) : QAbstractSpinBox(parent)
{
	m_minimum = std::numeric_limits<qlonglong>::min();
	m_maximum = std::numeric_limits<qlonglong>::max();
	m_value = 0;
	m_singleStep = 1;

	setValue(m_value);
}

qlonglong PVWidgets::PVLongLongSpinBox::value() const
{
	return m_value;
}

void PVWidgets::PVLongLongSpinBox::setValue(qlonglong expectedNewValue)
{
	const qlonglong newValue = qBound(m_minimum, expectedNewValue, m_maximum);
	const QString newValueString = QString::number(newValue);
	lineEdit()->setText(m_prefix + newValueString + m_suffix);
	if (m_value != newValue) {
		m_value = newValue;

		Q_EMIT valueChanged(newValue);
		Q_EMIT valueChanged(newValueString);
	}
}

QString PVWidgets::PVLongLongSpinBox::prefix() const
{
	return m_prefix;
}

void PVWidgets::PVLongLongSpinBox::setPrefix(const QString& prefix)
{
	m_prefix = prefix;

	setValue(m_value);
}

QString PVWidgets::PVLongLongSpinBox::suffix() const
{
	return m_suffix;
}

void PVWidgets::PVLongLongSpinBox::setSuffix(const QString& suffix)
{
	m_suffix = suffix;

	setValue(m_value);
}

QString PVWidgets::PVLongLongSpinBox::cleanText() const
{
	return QString::number(m_value);
}

qlonglong PVWidgets::PVLongLongSpinBox::singleStep() const
{
	return m_singleStep;
}

void PVWidgets::PVLongLongSpinBox::setSingleStep(qlonglong step)
{
	m_singleStep = step;
}

qlonglong PVWidgets::PVLongLongSpinBox::minimum() const
{
	return m_minimum;
}

void PVWidgets::PVLongLongSpinBox::setMinimum(qlonglong min)
{
	m_minimum = min;
	if (m_maximum < m_minimum) {
		m_maximum = m_minimum;
	}

	setValue(m_value);
}

qlonglong PVWidgets::PVLongLongSpinBox::maximum() const
{
	return m_maximum;
}

void PVWidgets::PVLongLongSpinBox::setMaximum(qlonglong max)
{
	m_maximum = max;
	if (m_maximum < m_minimum) {
		m_maximum = m_minimum;
	}

	setValue(m_value);
}

void PVWidgets::PVLongLongSpinBox::setRange(qlonglong min, qlonglong max)
{
	if (min < max) {
		m_minimum = min;
		m_maximum = max;
	} else {
		m_minimum = max;
		m_maximum = min;
	}

	setValue(m_value);
}

void PVWidgets::PVLongLongSpinBox::keyPressEvent(QKeyEvent* event)
{
	switch (event->key()) {
	case Qt::Key_Enter:
	case Qt::Key_Return:
		selectCleanText();
		lineEditEditingFinalize();
	}

	QAbstractSpinBox::keyPressEvent(event);
}

void PVWidgets::PVLongLongSpinBox::focusOutEvent(QFocusEvent* event)
{
	lineEditEditingFinalize();

	QAbstractSpinBox::focusOutEvent(event);
}

QAbstractSpinBox::StepEnabled PVWidgets::PVLongLongSpinBox::stepEnabled() const
{
	StepEnabled se;
	if (isReadOnly()) {
		return se;
	}

	if (wrapping() || m_value < m_maximum) {
		se |= StepUpEnabled;
	}
	if (wrapping() || m_value > m_minimum) {
		se |= StepDownEnabled;
	}

	return se;
}

void PVWidgets::PVLongLongSpinBox::stepBy(int steps)
{
	if (isReadOnly()) {
		return;
	}

	if (m_prefix + QString::number(m_value) + m_suffix != lineEdit()->text()) {
		lineEditEditingFinalize();
	}

	qlonglong newValue = m_value + (steps * m_singleStep);
	if (wrapping()) {
		// emulating the behavior of QSpinBox
		if (newValue > m_maximum) {
			if (m_value == m_maximum) {
				newValue = m_minimum;
			} else {
				newValue = m_maximum;
			}
		} else if (newValue < m_minimum) {
			if (m_value == m_minimum) {
				newValue = m_maximum;
			} else {
				newValue = m_minimum;
			}
		}
	} else {
		newValue = qBound(m_minimum, newValue, m_maximum);
	}

	setValue(newValue);
	selectCleanText();
}

QValidator::State PVWidgets::PVLongLongSpinBox::validate(QString& input, int& pos) const
{
	// first, we try to interpret as a number without prefixes
	bool ok;
	const qlonglong value = input.toLongLong(&ok);
	if (input.isEmpty() || (ok && value <= m_maximum)) {
		input = m_prefix + input + m_suffix;
		pos += m_prefix.length();
		return QValidator::Acceptable;
	}

	// if string of text editor aren't simple number, try to interpret it
	// as a number with prefix and suffix
	bool valid = true;
	if (!m_prefix.isEmpty() && !input.startsWith(m_prefix)) {
		valid = false;
	}
	if (!m_suffix.isEmpty() && !input.endsWith(m_suffix)) {
		valid = false;
	}

	if (valid) {
		const int start = m_prefix.length();
		const int length = input.length() - start - m_suffix.length();

		bool ok;
		const QString number = input.mid(start, length);
		const qlonglong value = number.toLongLong(&ok);
		if (number.isEmpty() || (ok && value <= m_maximum)) {
			return QValidator::Acceptable;
		}
	}

	// otherwise not acceptable
	return QValidator::Invalid;
}

void PVWidgets::PVLongLongSpinBox::lineEditEditingFinalize()
{
	const QString text = lineEdit()->text();

	// first, we try to read as a number without prefixes
	bool ok;
	qlonglong value = text.toLongLong(&ok);
	if (ok) {
		setValue(value);
		return;
	}

	// if string of text editor aren't simple number, try to interpret it
	// as a number with prefix and suffix
	bool valid = true;
	if (!m_prefix.isEmpty() && !text.startsWith(m_prefix)) {
		valid = false;
	} else if (!m_suffix.isEmpty() && !text.endsWith(m_suffix)) {
		valid = false;
	}

	if (valid) {
		const int start = m_prefix.length();
		const int length = text.length() - start - m_suffix.length();

		bool ok;
		const qlonglong value = text.mid(start, length).toLongLong(&ok);
		if (ok) {
			setValue(value);
			return;
		}
	}

	// otherwise set old value
	setValue(m_value);
}

void PVWidgets::PVLongLongSpinBox::selectCleanText()
{
	lineEdit()->setSelection(m_prefix.length(),
	                         lineEdit()->text().length() - m_prefix.length() - m_suffix.length());
}
