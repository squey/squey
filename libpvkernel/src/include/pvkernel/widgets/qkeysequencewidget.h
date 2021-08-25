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

#ifndef QKEYSEQUENCEWIDGET_H
#define QKEYSEQUENCEWIDGET_H

#include "qkeysequencewidget_p.h"

#include <QWidget>
#include <QIcon>

namespace PVWidgets
{
class QKeySequenceWidgetPrivate;

/*!
  \class QKeySequenceWidget

  \brief The QKeySequenceWidget is a widget to input a QKeySequence.

  This widget lets the user choose a QKeySequence, which is usually used as a
  shortcut key. The recording is initiated by calling captureKeySequence() or
  the user clicking into the widget.

  \code
    // create new QKeySequenceWidget with empty sequence
    QKeySequenceWidget *keyWidget = new QKeySequenceWidget;

    // Set sequence as "Ctrl+Alt+Space"
    keyWidget->setJeySequence(QKeySequence("Ctrl+Alt+Space"));

    // set clear button position is left
    setClearButtonShow(QKeySequenceWidget::ShowLeft);

    // set cutom clear button icon
    setClearButtonIcon(QIcon("/path/to/icon.png"));

    // connecting keySequenceChanged signal to slot
    connect(keyWidget, SIGNAL(keySequenceChanged(QKeySequence)), this,
  SLOT(slotKeySequenceChanged(QKeySequence)));
  \endcode
*/
class QKeySequenceWidget : public QWidget
{
	Q_OBJECT
	Q_DECLARE_PRIVATE(QKeySequenceWidget);
	Q_PRIVATE_SLOT(d_func(), void doneRecording())

	Q_PROPERTY(QKeySequence keySequence READ keySequence WRITE setKeySequence)
	Q_PROPERTY(QKeySequenceWidget::ClearButtonShow clearButton READ clearButtonShow WRITE
	               setClearButtonShow)
	Q_PROPERTY(QString noneText READ noneText WRITE setNoneText)
	Q_PROPERTY(QIcon clearButtonIcon READ clearButtonIcon WRITE setClearButtonIcon)

  private:
	QKeySequenceWidgetPrivate* const d_ptr;
	void _connectingSlots();

  private Q_SLOTS:
	void captureKeySequence();

  public:
	explicit QKeySequenceWidget(QWidget* parent = nullptr);
	explicit QKeySequenceWidget(QKeySequence seq, QWidget* parent = nullptr);
	explicit QKeySequenceWidget(QString noneString, QWidget* parent = nullptr);
	explicit QKeySequenceWidget(QKeySequence seq, QString noneString, QWidget* parent = nullptr);
	~QKeySequenceWidget() override;
	QSize sizeHint() const override;
	void setToolTip(const QString& tip);
	QKeySequence keySequence() const;
	QString noneText() const;
	QIcon clearButtonIcon() const;
	void setMaxNumKey(quint32 n);

	/*!
	  \brief Modes of sohow ClearButton
	*/
	enum ClearButton {
		NoShow = 0x00,   /**< Hide ClearButton */
		ShowLeft = 0x01, /**< ClearButton isow is left */
		ShowRight = 0x02 /**< ClearButton isow is left */
	};

	Q_DECLARE_FLAGS(ClearButtonShow, ClearButton);
	Q_FLAGS(ClearButtonShow)

	QKeySequenceWidget::ClearButtonShow clearButtonShow() const;

	static char get_ascii_from_sequence(QKeySequence key);

  Q_SIGNALS:
	void keySequenceChanged(const QKeySequence& seq);
	void keyNotSupported();

  public Q_SLOTS:
	void setKeySequence(const QKeySequence& key);
	void clearKeySequence();
	void setNoneText(const QString text);
	void setClearButtonIcon(const QIcon& icon);
	void setClearButtonShow(QKeySequenceWidget::ClearButtonShow show);

  private:
};
} // namespace PVWidgets

#endif // QKEYSEQUENCEWIDGET_H
