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

#ifndef __PVWIDGETS_PVLONGLONGSPINBOX_H__
#define __PVWIDGETS_PVLONGLONGSPINBOX_H__

#include <QAbstractSpinBox>
#include <QtGlobal>

namespace PVWidgets
{

class PVLongLongSpinBox : public QAbstractSpinBox
{
	Q_OBJECT
  public:
	explicit PVLongLongSpinBox(QWidget* parent = 0);

	qlonglong value() const;

	QString prefix() const;
	void setPrefix(const QString& prefix);

	QString suffix() const;
	void setSuffix(const QString& suffix);

	QString cleanText() const;

	qlonglong singleStep() const;
	void setSingleStep(qlonglong val);

	qlonglong minimum() const;
	void setMinimum(qlonglong min);

	qlonglong maximum() const;
	void setMaximum(qlonglong max);

	void setRange(qlonglong min, qlonglong max);

  public Q_SLOTS:
	void setValue(qlonglong value);

  Q_SIGNALS:
	void valueChanged(qlonglong i);
	void valueChanged(const QString& text);

  protected:
	virtual void keyPressEvent(QKeyEvent* event);
	virtual void focusOutEvent(QFocusEvent* event);
	virtual void stepBy(int steps);
	virtual StepEnabled stepEnabled() const;
	virtual QValidator::State validate(QString& input, int& pos) const;

  private:
	void lineEditEditingFinalize();
	void selectCleanText();

  private:
	QString m_prefix;
	QString m_suffix;
	qlonglong m_singleStep;
	qlonglong m_minimum;
	qlonglong m_maximum;
	qlonglong m_value;

  private:
	Q_DISABLE_COPY(PVLongLongSpinBox)
};

} // namespace PVWidgets

#endif // __PVWIDGETS_PVLONGLONGSPINBOX_H__
