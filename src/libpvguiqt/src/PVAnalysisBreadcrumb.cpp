/* * MIT License
 *
 * © Squey, 2026
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

#include <pvguiqt/PVAnalysisBreadcrumb.h>

#include <squey/PVAnalysisHistory.h>
#include <squey/PVRoot.h>

#include <QHBoxLayout>
#include <QLabel>
#include <QLocale>
#include <QScrollArea>
#include <QScrollBar>
#include <QStyle>
#include <QToolButton>

PVGuiQt::PVAnalysisBreadcrumb::PVAnalysisBreadcrumb(Squey::PVRoot& root, QWidget* parent)
    : QWidget(parent), _root(root)
{
	auto* layout = new QHBoxLayout(this);
	layout->setContentsMargins(4, 2, 4, 2);
	layout->setSpacing(2);

	/* Plain arrows rather than the icon set: it carries no rotate-right, and
	 * back and forward is what walking a trail means anyway.
	 */
	auto* undo_button = new QToolButton(this);
	undo_button->setIcon(style()->standardIcon(QStyle::SP_ArrowLeft));
	undo_button->setAutoRaise(true);
	undo_button->setToolTip(tr("Go back one step"));
	connect(undo_button, &QToolButton::clicked, this, &PVAnalysisBreadcrumb::undo);
	layout->addWidget(undo_button);

	auto* redo_button = new QToolButton(this);
	redo_button->setIcon(style()->standardIcon(QStyle::SP_ArrowRight));
	redo_button->setAutoRaise(true);
	redo_button->setToolTip(tr("Go forward one step"));
	connect(redo_button, &QToolButton::clicked, this, &PVAnalysisBreadcrumb::redo);
	layout->addWidget(redo_button);

	/* The whole trail is kept rather than elided: the history is bounded, and a
	 * step the user cannot see is a step they will not think of going back to.
	 */
	_strip = new QWidget();
	_strip_layout = new QHBoxLayout(_strip);
	_strip_layout->setContentsMargins(0, 0, 0, 0);
	_strip_layout->setSpacing(0);

	_scroll = new QScrollArea(this);
	_scroll->setWidget(_strip);
	_scroll->setWidgetResizable(true);
	_scroll->setFrameShape(QFrame::NoFrame);
	_scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
	_scroll->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	layout->addWidget(_scroll, 1);

	_history_changed = _root.history()._changed.connect(
	    sigc::mem_fun(*this, &PVAnalysisBreadcrumb::rebuild));

	rebuild();
}

PVGuiQt::PVAnalysisBreadcrumb::~PVAnalysisBreadcrumb()
{
	_history_changed.disconnect();
}

void PVGuiQt::PVAnalysisBreadcrumb::undo()
{
	_root.history().undo();
}

void PVGuiQt::PVAnalysisBreadcrumb::redo()
{
	_root.history().redo();
}

void PVGuiQt::PVAnalysisBreadcrumb::rebuild()
{
	while (QLayoutItem* item = _strip_layout->takeAt(0)) {
		delete item->widget();
		delete item;
	}

	const Squey::PVAnalysisHistory& history = _root.history();

	/* One step means the analysis has not moved yet, and a trail of one crumb
	 * is worth no room on screen.
	 */
	setVisible(history.size() > 1);
	if (history.size() <= 1) {
		Q_EMIT changed();
		return;
	}

	const size_t position = history.position();
	for (size_t i = 0; i < history.size(); i++) {
		add_crumb(i, i == position, i > position);
	}
	_strip_layout->addStretch(1);

	/* Landing anywhere may have scrolled the current step out of sight.
	 */
	if (QWidget* current = _strip_layout->itemAt(int(position) * 2)->widget()) {
		_scroll->ensureWidgetVisible(current);
	}

	Q_EMIT changed();
}

void PVGuiQt::PVAnalysisBreadcrumb::add_crumb(size_t index, bool is_current, bool is_ahead)
{
	const Squey::PVAnalysisStep& step = _root.history().step(index);

	if (index > 0) {
		auto* separator = new QLabel(QString::fromUtf8(" › "), _strip);
		separator->setEnabled(false);
		_strip_layout->addWidget(separator);
	} else {
		/* Kept so that every crumb sits at the same offset in the layout,
		 * which is how the current one is found again to be scrolled to.
		 */
		_strip_layout->addWidget(new QWidget(_strip));
	}

	auto* crumb = new QToolButton(_strip);
	crumb->setText(step.label());
	crumb->setAutoRaise(true);
	crumb->setToolTip(tr("%1 event(s) selected")
	                      .arg(QLocale().toString(qulonglong(step.selected_row_count()))));

	/* Where the user stands is set in bold, and what lies ahead -- the branch
	 * they walked back from, still there to walk forward into -- in italics.
	 * Both survive a change of theme, which a colour of our own would not.
	 */
	QFont font = crumb->font();
	font.setBold(is_current);
	font.setItalic(is_ahead);
	crumb->setFont(font);

	connect(crumb, &QToolButton::clicked, this,
	        [this, index] { _root.history().go_to(index); });

	_strip_layout->addWidget(crumb);
}
