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

#include <QDebug>
#include <QEvent>
#include <QKeyEvent>
#include <QIcon>

#include <pvkernel/widgets/qkeysequencewidget_p.h>
#include <pvkernel/widgets/qkeysequencewidget.h>

char PVWidgets::QKeySequenceWidget::get_ascii_from_sequence(QKeySequence key)
{
	// from qnamespace.h
	// Key_Escape = 0x01000000,
	// Key_Tab = 0x01000001,
	// Key_Backtab = 0x01000002,

	switch (key[0]) {
	case Qt::Key_Tab:
		return '\t';
	case Qt::Key_Backtab:
		return '\b';
	case Qt::Key_Escape:
		return 0x1b;
	case Qt::Key_Return:
	case Qt::Key_Enter:
		return 0x0a;
	default:
		return key[0];
	}
}

/*!
  Creates a QKeySequenceWidget object wuth \a parent and empty \a keySequence
*/
PVWidgets::QKeySequenceWidget::QKeySequenceWidget(QWidget* parent)
    : QWidget(parent), d_ptr(new PVWidgets::QKeySequenceWidgetPrivate())
{
	Q_D(PVWidgets::QKeySequenceWidget);
	d->q_ptr = this;
	d->init(QKeySequence(), QString());

	_connectingSlots();
}

/*!
  Creates a QKeySequenceWidget object wuth \a parent and keysequence \a keySequence
  and string for \a noneString
*/
PVWidgets::QKeySequenceWidget::QKeySequenceWidget(QKeySequence seq,
                                                  QString noneString,
                                                  QWidget* parent)
    : QWidget(parent), d_ptr(new PVWidgets::QKeySequenceWidgetPrivate())
{
	Q_D(PVWidgets::QKeySequenceWidget);
	d->q_ptr = this;
	qDebug() << "q_prt " << this;
	d->init(seq, noneString);
	_connectingSlots();
}

/*!
  Creates a QKeySequenceWidget object wuth \a parent and keysequence \a keySequence
*/
PVWidgets::QKeySequenceWidget::QKeySequenceWidget(QKeySequence seq, QWidget* parent)
    : QWidget(parent), d_ptr(new PVWidgets::QKeySequenceWidgetPrivate())
{
	qDebug() << "widget constructor";
	Q_D(QKeySequenceWidget);
	d->q_ptr = this;
	qDebug() << "q_prt " << this;
	d->init(seq, QString());
	_connectingSlots();
}

/*!
  Creates a QKeySequenceWidget object wuth \a parent and string for \a noneString
*/
PVWidgets::QKeySequenceWidget::QKeySequenceWidget(QString noneString, QWidget* parent)
    : QWidget(parent), d_ptr(new PVWidgets::QKeySequenceWidgetPrivate())
{
	qDebug() << "widget constructor";
	Q_D(PVWidgets::QKeySequenceWidget);
	d->q_ptr = this;
	qDebug() << "q_prt " << this;
	d->init(QKeySequence(), noneString);

	_connectingSlots();
}

/*!
  Destroy a QKeySequenceWidget object
*/
PVWidgets::QKeySequenceWidget::~QKeySequenceWidget()
{
	delete d_ptr;
}

QSize PVWidgets::QKeySequenceWidget::sizeHint() const
{
	return d_ptr->shortcutButton->sizeHint();
}

/*!
  Setting tooltip text to sequence button
  \param tip Text string
*/
void PVWidgets::QKeySequenceWidget::setToolTip(const QString& tip)
{
	d_ptr->setToolTip(tip);
}

/*!
  Setting mode of Clear Buttorn display.
  \param show Position of clear button \a ClearButtornShow
  \sa clearButtonShow
*/
void PVWidgets::QKeySequenceWidget::setClearButtonShow(
    PVWidgets::QKeySequenceWidget::ClearButtonShow show)
{
	d_ptr->showClearButton = show;
	d_ptr->updateView();
}

/*!
  Return mode of clear button dosplay.
  \param show Display mode of clear button (NoShow, ShowLeft or ShorRight)
  \sa setClearButtonShow
*/
PVWidgets::QKeySequenceWidget::ClearButtonShow
PVWidgets::QKeySequenceWidget::clearButtonShow() const
{
	return static_cast<PVWidgets::QKeySequenceWidget::ClearButtonShow>(d_ptr->showClearButton);
}

/*!
    Set the key sequence.
    \param key Key sequence
    \sa clearKeySequence
 */
void PVWidgets::QKeySequenceWidget::setKeySequence(const QKeySequence& key)
{
	if (d_ptr->isRecording == false) {
		d_ptr->oldSequence = d_ptr->currentSequence;
	}

	d_ptr->doneRecording();

	d_ptr->currentSequence = key;
	d_ptr->doneRecording();
}

/*!
    Get current key sequence.
    \return Current key sequence
    \sa setKeySequence
    \sa clearKeySequence
 */
QKeySequence PVWidgets::QKeySequenceWidget::keySequence() const
{
	return d_ptr->currentSequence;
}

/*!
    Clear key sequence.
    \sa setKeySequence
 */
void PVWidgets::QKeySequenceWidget::clearKeySequence()
{
	setKeySequence(QKeySequence());
}

// slot for capture key sequence starting (private)
void PVWidgets::QKeySequenceWidget::captureKeySequence()
{
	d_ptr->startRecording();
}

/*!
    Set string for display when key sequence is undefined.
    \param text Text string
    \sa noneText
 */
void PVWidgets::QKeySequenceWidget::setNoneText(const QString text)
{
	d_ptr->noneSequenceText = text;
	d_ptr->updateDisplayShortcut();
}

/*!
    Get string for display when key sequence is undefined.
    \return Text string
    \sa setNoneText
 */
QString PVWidgets::QKeySequenceWidget::noneText() const
{
	return d_ptr->noneSequenceText;
}

/*!
    Set custom icon for clear buttom.
    \param icon QIcon object
    \sa clearButtonIcon
 */
void PVWidgets::QKeySequenceWidget::setClearButtonIcon(const QIcon& icon)
{
	d_ptr->clearButton->setIcon(icon);
}

/*!
    Get clear buttom icon.
    \return QIcon object
    \sa setClearButtonIcon
 */
QIcon PVWidgets::QKeySequenceWidget::clearButtonIcon() const
{
	return d_ptr->clearButton->icon();
}

void PVWidgets::QKeySequenceWidget::setMaxNumKey(quint32 n)
{
	if (n < 1) {
		n = 1;
	}
	d_ptr->maxNumKey = n;
}

// connection internal signals & slots
void PVWidgets::QKeySequenceWidget::_connectingSlots()
{
	// connect signals to slots
	connect(d_ptr->clearButton, &QAbstractButton::clicked, this,
	        &QKeySequenceWidget::clearKeySequence);
	connect(&d_ptr->modifierlessTimeout, SIGNAL(timeout()), this, SLOT(doneRecording()));
	connect(d_func()->shortcutButton, &QAbstractButton::clicked, this,
	        &QKeySequenceWidget::captureKeySequence);
}

// Private class implementation

PVWidgets::QKeySequenceWidgetPrivate::QKeySequenceWidgetPrivate()
    : maxNumKey(4), layout(nullptr), clearButton(nullptr), shortcutButton(nullptr)
{
	Q_Q(PVWidgets::QKeySequenceWidget);
	Q_UNUSED(q);
}

PVWidgets::QKeySequenceWidgetPrivate::~QKeySequenceWidgetPrivate() = default;

void PVWidgets::QKeySequenceWidgetPrivate::init(const QKeySequence keySeq, const QString noneStr)
{
	Q_Q(PVWidgets::QKeySequenceWidget);
	Q_UNUSED(q);
	layout = new QHBoxLayout(q_func());
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(1);

	clearButton = new QToolButton(q_func());
	clearButton->setText("x");

	layout->addWidget(clearButton);

	shortcutButton = new QShortcutButton(this, q_func());

	if (noneStr.isNull() == true) {
		noneSequenceText = "...";
	} else {
		noneSequenceText = noneStr;
	}

	q_ptr->clearKeySequence();
	currentSequence = keySeq;

	shortcutButton->setFocusPolicy(Qt::StrongFocus);

	layout->addWidget(shortcutButton);

	showClearButton = PVWidgets::QKeySequenceWidget::ShowRight;

	clearButton->setIcon(QIcon(":/img/delete_32.png"));

	// unfocused clear button  afyer created (small hack)
	clearButton->setFocusPolicy(Qt::NoFocus);

	// update ui
	updateDisplayShortcut();
	updateView();
}

// set tooltip only for seqyence button
void PVWidgets::QKeySequenceWidgetPrivate::setToolTip(const QString& tip)
{
	shortcutButton->setToolTip(tip);
	clearButton->setToolTip("");
}

// update the location of widgets
void PVWidgets::QKeySequenceWidgetPrivate::updateView()
{
	// qDebug() << "update view ";
	switch (showClearButton) {
	case PVWidgets::QKeySequenceWidget::ShowLeft:
		clearButton->setVisible(true);
		layout->setDirection(QBoxLayout::LeftToRight);
		break;
	case PVWidgets::QKeySequenceWidget::ShowRight:
		clearButton->setVisible(true);
		layout->setDirection(QBoxLayout::RightToLeft);
		break;
	case PVWidgets::QKeySequenceWidget::NoShow:
		clearButton->setVisible(false);
		break;
	default:
		layout->setDirection(QBoxLayout::LeftToRight);
	}
}

void PVWidgets::QKeySequenceWidgetPrivate::startRecording()
{
	numKey = 0;
	modifierKeys = 0;
	oldSequence = currentSequence;
	currentSequence = QKeySequence();
	isRecording = true;
	shortcutButton->setDown(true);

	shortcutButton->grabKeyboard();

	if (!QWidget::keyboardGrabber()) {
		qWarning() << "Failed to grab the keyboard! Most likely qt's nograb option is active";
	}

	// update Shortcut display
	updateDisplayShortcut();
}

void PVWidgets::QKeySequenceWidgetPrivate::doneRecording()
{
	modifierlessTimeout.stop();

	isRecording = false;
	shortcutButton->releaseKeyboard();
	shortcutButton->setDown(false);

	// if sequence is not changed
	if (currentSequence == oldSequence) {
		// update Shortcut display
		updateDisplayShortcut();

		return;
	}

	// key sequnce is changed
	Q_EMIT q_ptr->keySequenceChanged(currentSequence);

	// update Shortcut display
	updateDisplayShortcut();
}

inline void PVWidgets::QKeySequenceWidgetPrivate::cancelRecording()
{
	currentSequence = oldSequence;
	doneRecording();
}

inline void PVWidgets::QKeySequenceWidgetPrivate::controlModifierlessTimout()
{
	if (numKey != 0 && !modifierKeys) {
		// No modifier key pressed currently. Start the timout
		modifierlessTimeout.start(600);
	} else {
		// A modifier is pressed. Stop the timeout
		modifierlessTimeout.stop();
	}
}

inline void PVWidgets::QKeySequenceWidgetPrivate::keyNotSupported()
{
	Q_EMIT q_ptr->keyNotSupported();
}

void PVWidgets::QKeySequenceWidgetPrivate::updateDisplayShortcut()
{
	// empty string if no non-modifier was pressed
	QString str = currentSequence.toString(QKeySequence::NativeText);
	str.replace('&', QLatin1String("&&")); // TODO -- check it

	if (isRecording == true) {
		if (modifierKeys) {
			if (str.isEmpty() == false)
				str.append(",");

			if ((modifierKeys & Qt::META))
				str += "Meta + ";

			if ((modifierKeys & Qt::CTRL))
				str += "Ctrl + ";

			if ((modifierKeys & Qt::ALT))
				str += "Alt + ";

			if ((modifierKeys & Qt::SHIFT))
				str += "Shift + ";
		}

		// make it clear that input is still going on
		str.append("...");
	}

	// if is noting
	if (str.isEmpty() == true) {
		str = noneSequenceText;
	}

	// if it is Tab
	if (str == "\t") {
		str = "Tab";
	}

	shortcutButton->setText(str);
}

// QKeySequenceButton implementation
QSize PVWidgets::QShortcutButton::sizeHint() const
{
	return QPushButton::sizeHint();
}

bool PVWidgets::QShortcutButton::event(QEvent* e)
{
	if (d->isRecording == true && e->type() == QEvent::KeyPress) {
		keyPressEvent(static_cast<QKeyEvent*>(e));
		return true;
	}

	if (d->isRecording && e->type() == QEvent::ShortcutOverride) {
		e->accept();
		return true;
	}

	if (d->isRecording == true && e->type() == QEvent::FocusOut) {
		d->cancelRecording();
		return true;
	}

	return QPushButton::event(e);
}

void PVWidgets::QShortcutButton::keyPressEvent(QKeyEvent* keyEvent)
{
	// qDebug() << "key pressed";
	int keyQt = keyEvent->key();

	// Qt sometimes returns garbage keycodes, I observed -1,
	// if it doesn't know a key.
	// We cannot do anything useful with those (several keys have -1,
	// indistinguishable)
	// and QKeySequence.toString() will also yield a garbage string.
	if (keyQt == -1) {
		// keu moy supported in Qt
		d->cancelRecording();
		d->keyNotSupported();
	}

	// get modifiers key
	uint newModifiers = keyEvent->modifiers() & (Qt::SHIFT | Qt::CTRL | Qt::ALT | Qt::META);

	// block autostart capturing on key_return or key space press
	if (d->isRecording == false && (keyQt == Qt::Key_Return || keyQt == Qt::Key_Space)) {
		return;
	}

	// We get events even if recording isn't active.
	if (d->isRecording == false) {
		return QPushButton::keyPressEvent(keyEvent);
	}

	keyEvent->accept();
	d->modifierKeys = newModifiers;

	// switching key type
	switch (keyQt) {
	case Qt::Key_AltGr: // or else we get unicode salad
		return;
	case Qt::Key_Shift:
	case Qt::Key_Control:
	case Qt::Key_Alt:
	case Qt::Key_Meta:
	case Qt::Key_Menu: // unused (yes, but why?)
		// TODO - check it key
		d->controlModifierlessTimout();
		d->updateDisplayShortcut();
		break;
	default: {
	}

		// We now have a valid key press.
		if (keyQt) {
			if ((keyQt == Qt::Key_Backtab) && (d->modifierKeys & Qt::SHIFT)) {
				keyQt = Qt::Key_Tab | d->modifierKeys;
			} else // if (d->isShiftAsModifierAllowed(keyQt))
			{
				keyQt |= d->modifierKeys;
			}

			if (d->numKey == 0) {
				d->currentSequence = QKeySequence(keyQt);
			}

			d->numKey++; // increment nuber of pressed keys

			if (d->numKey >= d->maxNumKey) {
				d->doneRecording();
				return;
			}

			d->controlModifierlessTimout();
			d->updateDisplayShortcut();
		}
	}
}

void PVWidgets::QShortcutButton::keyReleaseEvent(QKeyEvent* keyEvent)
{
	// qDebug() << "key released";
	if (keyEvent->key() == -1) {
		// ignore garbage, see keyPressEvent()
		return;
	}

	// if not recording mode
	if (d->isRecording == false) {
		return QPushButton::keyReleaseEvent(keyEvent);
	}

	keyEvent->accept();

	uint newModifiers = keyEvent->modifiers() & (Qt::SHIFT | Qt::CTRL | Qt::ALT | Qt::META);

	// if a modifier that belongs to the shortcut was released...
	if ((newModifiers & d->modifierKeys) < d->modifierKeys) {
		d->modifierKeys = newModifiers;
		d->controlModifierlessTimout();
		d->updateDisplayShortcut();
	}
}
