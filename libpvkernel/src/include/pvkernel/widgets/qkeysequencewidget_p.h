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

#ifndef QKEYSEQUENCEWIDGET_P_H
#define QKEYSEQUENCEWIDGET_P_H

#include <QKeySequence>
#include <QHBoxLayout>
#include <QToolButton>
#include <QPushButton>
#include <QTimer>
#include <QDebug>
#include <QIcon>

#include "qkeysequencewidget.h"

namespace PVWidgets
{

class QShortcutButton;
class QKeySequenceWidget;

class QKeySequenceWidgetPrivate // : public QObject
{
	// Q_OBJECT
	Q_DECLARE_PUBLIC(QKeySequenceWidget);

  public:
	QKeySequenceWidget* q_ptr;

	QKeySequenceWidgetPrivate();
	virtual ~QKeySequenceWidgetPrivate();

	void init(const QKeySequence keySeq, const QString noneStr);
	void updateView();

	void startRecording();
	void doneRecording();
	inline void cancelRecording();
	inline void controlModifierlessTimout();
	inline void keyNotSupported();

	void updateDisplayShortcut();

	// members
	QKeySequence currentSequence;
	QKeySequence oldSequence;
	QString noneSequenceText;

	QTimer modifierlessTimeout;

	quint32 numKey;
	quint32 maxNumKey;
	quint32 modifierKeys;

	void setToolTip(const QString& tip);

	QHBoxLayout* layout;
	QToolButton* clearButton;
	QShortcutButton* shortcutButton;

	int showClearButton;

	bool isRecording;
};

class QShortcutButton : public QPushButton
{
	Q_OBJECT

  public:
	explicit QShortcutButton(QKeySequenceWidgetPrivate* p, QWidget* parent = nullptr)
	    : QPushButton(parent), d(p)
	{
		/* qDebug() << "qShortcut button Create"; */
		/* qDebug() << "parent----" << parent; */

		/* qDebug() << "visible " << isVisible();       */
		setMinimumWidth(QPushButton::minimumWidth());
		QPushButton::setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
	}

	~QShortcutButton() override = default;

	QSize sizeHint() const override;

  protected:
	// Reimplemented for internal reasons.
	bool event(QEvent* e) override;
	void keyPressEvent(QKeyEvent* keyEvent) override;
	void keyReleaseEvent(QKeyEvent* keyEvent) override;

  private:
	QKeySequenceWidgetPrivate* const d;
};
} // namespace PVWidgets

#endif // QKEYSEQUENCEWIDGET_P_H
