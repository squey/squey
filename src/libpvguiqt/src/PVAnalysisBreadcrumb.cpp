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
#include <QToolButton>

/* One row of small icons, meant to cost as little height as a toolbar does.
 *
 * Only the icon sizes are ours. How much room a button needs around one is the
 * style's business, and the style here is a stylesheet, which no button knows
 * about until it is in the window that carries it: a button asked beforehand
 * answers for a style nobody uses. So nothing is pinned to a number -- the row
 * takes its height from the buttons it actually holds, every time it is built.
 */
static constexpr int icon_pixels = 16;

/* Narrower than the crumbs, and deliberately so: it separates them rather than
 * standing among them.
 */
static constexpr int chevron_pixels = 10;

namespace
{

/* A scroll area is as tall as it feels like being, which for one row of buttons
 * is three times too much. Saying so once, here, is what keeps the row honest:
 * setting a height by hand instead meant setting it again whenever the style or
 * the contents moved, and each of those adjustments moved the row.
 */
class PVRowScrollArea : public QScrollArea
{
  public:
	using QScrollArea::QScrollArea;

	QSize sizeHint() const override
	{
		const QSize inner = widget() != nullptr ? widget()->sizeHint() : QSize();
		return QSize(inner.width(), inner.height() + 2 * frameWidth());
	}

	/**
	 * The height the row needs, and no width at all.
	 *
	 * Only the height is worth insisting on: it is what keeps the row from being
	 * squeezed under its own contents. Asking for the width as well made the
	 * window unable to be narrower than the whole trail, so every step taken
	 * widened it -- which is the one thing a scroll area is there to avoid.
	 */
	QSize minimumSizeHint() const override { return QSize(0, sizeHint().height()); }
};

} // namespace

PVGuiQt::PVAnalysisBreadcrumb::PVAnalysisBreadcrumb(Squey::PVRoot& root, QWidget* parent)
    : QWidget(parent), _root(root)
{
	auto* layout = new QHBoxLayout(this);
	layout->setContentsMargins(2, 0, 2, 0);
	layout->setSpacing(0);

	/* From the icon set like the crumbs, and not from the style: two buttons
	 * carrying icons of different kinds do not measure the same, and the row
	 * would then change height the moment a first crumb joined the arrows.
	 * Arrows with a shaft rather than the solid triangles the set uses for
	 * spin boxes and scroll bars, and rather than rotate-left and rotate-right,
	 * of which it has only the first -- back and forward is what walking a
	 * trail means anyway.
	 */
	_undo_button = make_button(PVModdedIcon("arrow-left-long"));
	connect(_undo_button, &QToolButton::clicked, this, &PVAnalysisBreadcrumb::undo);
	layout->addWidget(_undo_button);

	_redo_button = make_button(PVModdedIcon("arrow-right-long"));
	connect(_redo_button, &QToolButton::clicked, this, &PVAnalysisBreadcrumb::redo);
	layout->addWidget(_redo_button);

	/* The whole trail is kept rather than elided: the history is bounded, and a
	 * step nobody can see is a step nobody will think of going back to.
	 */
	_strip = new QWidget();
	_strip_layout = new QHBoxLayout(_strip);
	_strip_layout->setContentsMargins(0, 0, 0, 0);
	_strip_layout->setSpacing(0);

	_scroll = new PVRowScrollArea(this);
	_scroll->setWidget(_strip);
	_scroll->setWidgetResizable(true);
	_scroll->setFrameShape(QFrame::NoFrame);

	/* NoFrame is not enough: the stylesheet frames scroll areas, and it is the
	 * stylesheet that frameWidth() reports. Those three pixels a side are six
	 * pixels of height the row gains the moment the strip holds anything, which
	 * is what made it grow after the first step -- and, drawn, they are what
	 * clipped the icons. This one is a layout device, not a view to be framed.
	 */
	_scroll->setStyleSheet("QScrollArea { border: none; padding: 0; }");
	_scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	_scroll->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	layout->addWidget(_scroll, 1);

	setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);

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

	/* A stylesheet reaches a button when the button is polished, and a button
	 * is polished after it has been created -- possibly after somebody asked it
	 * how big it wanted to be. Ask for the polish here and then say the sizes
	 * moved, or the row is laid out from what the buttons measured before the
	 * style reached them, and only settles on the rebuild after.
	 */
	_strip->ensurePolished();
	_strip->updateGeometry();
	_scroll->updateGeometry();
	updateGeometry();

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
		auto* chevron = new PVModdedIconLabel("branch-closed", QSize(chevron_pixels, chevron_pixels));
		chevron->setParent(_strip);
		chevron->setEnabled(false);
		chevron->setAlignment(Qt::AlignCenter);
		/* Sized both ways, not just across: a label is free to ask for the
		 * height a stylesheet gives labels, and one asking for more than the
		 * buttons around it is what made the row grow the moment a first
		 * chevron joined the arrows.
		 */
		chevron->setFixedSize(chevron_pixels, chevron_pixels);
		_strip_layout->addWidget(chevron);
	}

	auto* crumb = new QToolButton(_strip);
	crumb->setIcon(PVModdedIcon(QString::fromStdString(step.icon())));
	crumb->setIconSize(QSize(icon_pixels, icon_pixels));
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
