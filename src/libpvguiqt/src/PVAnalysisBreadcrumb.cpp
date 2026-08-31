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

#include <pvkernel/widgets/PVModdedIcon.h>

#include <QGraphicsOpacityEffect>
#include <QHBoxLayout>
#include <QLocale>
#include <QScrollArea>
#include <QScrollBar>
#include <QStyle>
#include <QToolButton>

/* One row of small icons, meant to cost as little height as a toolbar does.
 * Only the icon size is ours: how much room a button needs around one is the
 * style's business. Hardcoding that room is how the icons ended up cut off, and
 * leaving the row unconstrained is how it grew to three times its need -- so
 * ask a button of the very shape used here, once, and hold everything to that.
 */
static constexpr int icon_pixels = 16;

static int row_height()
{
	QToolButton probe;

	probe.setIconSize(QSize(icon_pixels, icon_pixels));
	probe.setAutoRaise(true);

	return probe.sizeHint().height();
}

PVGuiQt::PVAnalysisBreadcrumb::PVAnalysisBreadcrumb(Squey::PVRoot& root, QWidget* parent)
    : QWidget(parent), _root(root)
{
	_row_pixels = row_height();

	auto* layout = new QHBoxLayout(this);
	layout->setContentsMargins(2, 0, 2, 0);
	layout->setSpacing(0);

	/* The style's own arrows rather than the icon set: it carries no
	 * rotate-right, and back and forward is what walking a trail means anyway.
	 */
	_undo_button = make_button(style()->standardIcon(QStyle::SP_ArrowLeft));
	connect(_undo_button, &QToolButton::clicked, this, &PVAnalysisBreadcrumb::undo);
	layout->addWidget(_undo_button);

	_redo_button = make_button(style()->standardIcon(QStyle::SP_ArrowRight));
	connect(_redo_button, &QToolButton::clicked, this, &PVAnalysisBreadcrumb::redo);
	layout->addWidget(_redo_button);

	/* The whole trail is kept rather than elided: the history is bounded, and a
	 * step nobody can see is a step nobody will think of going back to.
	 */
	_strip = new QWidget();
	_strip_layout = new QHBoxLayout(_strip);
	_strip_layout->setContentsMargins(0, 0, 0, 0);
	_strip_layout->setSpacing(0);

	_scroll = new QScrollArea(this);
	_scroll->setWidget(_strip);
	_scroll->setWidgetResizable(true);
	_scroll->setFrameShape(QFrame::NoFrame);
	_scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	_scroll->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	_scroll->setFixedHeight(_row_pixels);
	layout->addWidget(_scroll, 1);

	setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
	setFixedHeight(_row_pixels);

	_history_changed =
	    _root.history()._changed.connect(sigc::mem_fun(*this, &PVAnalysisBreadcrumb::rebuild));

	rebuild();
}

PVGuiQt::PVAnalysisBreadcrumb::~PVAnalysisBreadcrumb()
{
	_history_changed.disconnect();
}

QToolButton* PVGuiQt::PVAnalysisBreadcrumb::make_button(const QIcon& icon)
{
	auto* button = new QToolButton(this);

	button->setIcon(icon);
	button->setIconSize(QSize(icon_pixels, icon_pixels));
	button->setFixedSize(_row_pixels, _row_pixels);
	button->setAutoRaise(true);
	button->setFocusPolicy(Qt::NoFocus);

	return button;
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

	_undo_button->setEnabled(history.can_undo());
	_redo_button->setEnabled(history.can_redo());

	/* One step means the analysis has not moved yet. The row stays where it is
	 * rather than coming and going: a toolbar that appears has to be laid out
	 * as it appears, and the first crumbs showed up at the wrong height for it.
	 * Two arrows greyed out say "nothing to come back to" well enough.
	 */
	if (history.size() <= 1) {
		Q_EMIT changed();
		return;
	}

	const size_t position = history.position();
	QWidget* current = nullptr;
	for (size_t i = 0; i < history.size(); i++) {
		QWidget* crumb = add_crumb(i, i == position, i > position);
		if (i == position) {
			current = crumb;
		}
	}
	_strip_layout->addStretch(1);

	/* Landing anywhere may have left the current step off the visible part of
	 * the strip.
	 */
	if (current != nullptr) {
		_scroll->ensureWidgetVisible(current);
	}

	Q_EMIT changed();
}

QWidget* PVGuiQt::PVAnalysisBreadcrumb::add_crumb(size_t index, bool is_current, bool is_ahead)
{
	const Squey::PVAnalysisStep& step = _root.history().step(index);

	/* A chevron between crumbs, so that the row reads as a trail rather than as
	 * a handful of buttons that happen to sit side by side. Drawn from the icon
	 * set like everything else here, so that it follows the theme rather than
	 * the text colour.
	 */
	if (index > 0) {
		auto* chevron = new PVModdedIconLabel("branch-closed", QSize(icon_pixels, icon_pixels));
		chevron->setParent(_strip);
		chevron->setEnabled(false);
		chevron->setAlignment(Qt::AlignCenter);
		_strip_layout->addWidget(chevron);
	}

	auto* crumb = new QToolButton(_strip);
	crumb->setIcon(PVModdedIcon(QString::fromStdString(step.icon())));
	crumb->setIconSize(QSize(icon_pixels, icon_pixels));
	crumb->setFixedSize(_row_pixels, _row_pixels);
	crumb->setAutoRaise(true);
	crumb->setFocusPolicy(Qt::NoFocus);
	crumb->setToolTip(tr("%1\n%2 event(s) selected")
	                      .arg(step.label())
	                      .arg(QLocale().toString(qulonglong(step.selected_row_count()))));

	/* Where the user stands is the one pressed in; what lies ahead -- the
	 * branch they walked back from, still there to walk forward into -- is
	 * faded. Both come from the style, so both survive a change of theme.
	 */
	crumb->setCheckable(true);
	crumb->setChecked(is_current);
	if (is_ahead) {
		auto* faded = new QGraphicsOpacityEffect(crumb);
		faded->setOpacity(0.4);
		crumb->setGraphicsEffect(faded);
	}

	connect(crumb, &QToolButton::clicked, this, [this, index] { _root.history().go_to(index); });

	_strip_layout->addWidget(crumb);

	return crumb;
}
