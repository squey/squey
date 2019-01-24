/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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
