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
#include <QPointer>
#include <QTimer>
#include <QScrollArea>
#include <QScrollBar>
#include <QWheelEvent>
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

	/**
	 * Walk the trail with the wheel.
	 *
	 * The scroll bars are kept out of sight -- one would add its own height to a
	 * row whose height is the whole point of measuring it here -- so without this
	 * a trail longer than the window has steps that nothing can reach. Either
	 * axis of the wheel walks it, this row running only one way.
	 */
	void wheelEvent(QWheelEvent* event) override
	{
		QScrollBar* const bar = horizontalScrollBar();
		if (bar == nullptr) {
			QScrollArea::wheelEvent(event);
			return;
		}

		const QPoint pixels = event->pixelDelta();
		const QPoint degrees = event->angleDelta();
		const int by = not pixels.isNull() ? pixels.x() + pixels.y()
		                                   : (degrees.x() + degrees.y()) / 2;

		bar->setValue(bar->value() - by);
		event->accept();
	}
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

	/* Set apart from the trail: the arrows walk it, they are not part of it.
	 */
	auto* divider = new PVModdedIconLabel("line-vertical", QSize(icon_pixels, icon_pixels));
	divider->setParent(this);
	divider->setEnabled(false);
	divider->setAlignment(Qt::AlignCenter);
	divider->setFixedWidth(chevron_pixels);
	layout->addWidget(divider);
	_divider = divider;

	/* The whole trail is kept rather than elided: the history is bounded, and a
	 * step nobody can see is a step nobody will think of going back to.
	 */
	_strip = new QWidget();
	_strip_layout = new QHBoxLayout(_strip);
	_strip_layout->setContentsMargins(0, 0, 0, 0);
	_strip_layout->setSpacing(0);

	// Packs the trail to the left, and stays the last item of the layout: crumbs
	// are inserted before it as the history grows.
	_strip_layout->addStretch(1);

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
	/* Going to the end of the trail has to wait for the scroll area to have
	 * measured the strip and worked out how far it can scroll, which it does on
	 * its own schedule -- after the strip's layout, after the widget has been
	 * resized. Rather than guess when that is, the bar says so: it announces its
	 * new range, and that is when the end is known.
	 */
	connect(_scroll->horizontalScrollBar(), &QScrollBar::rangeChanged, this,
	        [this](int /*min*/, int max) {
		        if (_follow_end) {
			        _follow_end = false;
			        _scroll->horizontalScrollBar()->setValue(max);
		        }
	        });

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
	const Squey::PVAnalysisHistory& history = _root.history();


	_undo_button->setEnabled(history.can_undo());
	_redo_button->setEnabled(history.can_redo());

	/* There is nothing to set the arrows apart from until a trail exists. The
	 * layout takes the room back, this being a plain child rather than
	 * something a toolbar is holding.
	 */
	_divider->setVisible(history.size() > 1);

	/* One step means the analysis has not moved yet. The row stays where it is
	 * rather than coming and going: a toolbar that appears has to be laid out
	 * as it appears, and the first crumbs showed up at the wrong height for it.
	 * Two arrows greyed out say "nothing to come back to" well enough.
	 */
	const size_t wanted = history.size() > 1 ? history.size() : 0;

	/* The crumbs standing are kept and told what they now say. Building the trail
	 * again from scratch meant every crumb was destroyed and remade whenever one
	 * was added -- the whole row blinking for a step that only ever appends at
	 * its end -- and left them without a position long enough that asking to
	 * bring one into view scrolled back to the first.
	 *
	 * So only the difference is built: crumbs beyond what the history holds are
	 * let go of, missing ones are added at the end, and the rest stay exactly
	 * where they are.
	 */
	while (_crumbs.size() > wanted) {
		delete _crumbs.back();
		_crumbs.pop_back();
		if (not _chevrons.empty()) {
			delete _chevrons.back();
			_chevrons.pop_back();
		}
	}

	while (_crumbs.size() < wanted) {
		const size_t index = _crumbs.size();

		/* A chevron between crumbs, so that the row reads as a trail rather than
		 * as a handful of buttons that happen to sit side by side. Drawn from the
		 * icon set like everything else here, so that it follows the theme rather
		 * than the text colour.
		 */
		if (index > 0) {
			auto* chevron =
			    new PVModdedIconLabel("branch-closed", QSize(chevron_pixels, chevron_pixels));
			chevron->setParent(_strip);
			chevron->setEnabled(false);
			chevron->setAlignment(Qt::AlignCenter);
			/* Sized both ways, not just across: a label is free to ask for the
			 * height a stylesheet gives labels, and one asking for more than the
			 * buttons around it is what made the row grow the moment a first
			 * chevron joined the arrows.
			 */
			chevron->setFixedSize(chevron_pixels, chevron_pixels);
			_strip_layout->insertWidget(_strip_layout->count() - 1, chevron);
			_chevrons.push_back(chevron);
		}

		auto* crumb = new QToolButton(_strip);
		crumb->setIconSize(QSize(icon_pixels, icon_pixels));
		crumb->setAutoRaise(true);
		crumb->setFocusPolicy(Qt::NoFocus);
		crumb->setCheckable(true);
		connect(crumb, &QToolButton::clicked, this, [this, crumb]() {
			const auto found = std::find(_crumbs.begin(), _crumbs.end(), crumb);
			if (found != _crumbs.end()) {
				_root.history().go_to(size_t(found - _crumbs.begin()));
			}
		});

		// Before the trailing stretch, which keeps the trail packed to the left.
		_strip_layout->insertWidget(_strip_layout->count() - 1, crumb);
		_crumbs.push_back(crumb);
	}

	if (wanted == 0) {
		Q_EMIT changed();
		return;
	}

	const size_t position = history.position();
	for (size_t i = 0; i < _crumbs.size(); i++) {
		refresh_crumb(_crumbs[i], i, i == position, i > position);
	}

	/* Landing anywhere may have left the current step off the visible part of
	 * the strip. Asked for once the scroll area has laid the strip out, which it
	 * does on coming back to its event loop: a crumb added a moment ago has no
	 * place yet, and asking now answers with the strip's own origin -- which
	 * scrolls back to the first crumb rather than to the one landed on.
	 */
	/* Standing on the last step means the end of the trail, which is where the
	 * crumb just added is. Asked for by the flag above rather than by scrolling
	 * now: how far the strip reaches is not settled at this point, and every
	 * attempt at it landed on the crumb before the last.
	 */
	_follow_end = position + 1 == _crumbs.size();

	QPointer<QWidget> target = _crumbs[position];
	QTimer::singleShot(0, this, [this, target]() {
		if (target == nullptr or _follow_end) {
			// The end of the trail is being gone to, which covers this crumb.
			return;
		}

		// Coming back to a step in the middle: the layout is settled by now.
		_strip_layout->activate();
		_scroll->ensureWidgetVisible(target);
	});

	Q_EMIT changed();
}

void PVGuiQt::PVAnalysisBreadcrumb::refresh_crumb(QToolButton* crumb,
                                                  size_t index,
                                                  bool is_current,
                                                  bool is_ahead)
{
	const Squey::PVAnalysisStep& step = _root.history().step(index);

	crumb->setIcon(PVModdedIcon(QString::fromStdString(step.icon())));

	/* Three things a crumb has to say -- what it was, what it was given, what it
	 * left selected -- ruled off from one another. Rich text, so what is quoted
	 * has to be escaped: a query saying "col1 < 2" would otherwise lose
	 * everything from the angle bracket on.
	 */
	auto block = [](QString const& text) {
		return QString("<div align=\"center\">%1</div>")
		    .arg(QString(text).toHtmlEscaped().replace(QChar('\n'), "<br/>"));
	};

	/* Drawn rather than ruled: <hr/> is a raised border whose weight Qt gives no
	 * way of setting, and next to two short lines of text it reads as a divider
	 * between two tooltips rather than between two parts of one.
	 */
	const QString rule = block(QString(QChar(0x2500)).repeated(12));

	QString tip = block(step.label());
	if (not step.details().isEmpty()) {
		tip += rule + block(step.details());
	}
	/* In English, as the parallel view counts its rows: the application runs
	 * under the C locale, which groups no digits.
	 */
	const QLocale english(QLocale::English);
	const QString selected = english.toString(qulonglong(step.selected_row_count()));
	tip += rule + block(step.valid_row_count() == 0
	                        ? tr("%1 event(s) selected").arg(selected)
	                        : tr("%1 event(s) selected (%2%)")
	                              .arg(selected,
	                                   english.toString(100. * double(step.selected_row_count()) /
	                                                        double(step.valid_row_count()),
	                                                    'f', 1)));

	crumb->setToolTip(tip);

	/* Where the user stands is the one pressed in; what lies ahead -- the
	 * branch they walked back from, still there to walk forward into -- is
	 * faded. Both come from the style, so both survive a change of theme.
	 *
	 * The effect is set and unset rather than made anew, a crumb outliving the
	 * step it stood for: one left over from a previous position would go on
	 * being faded after the trail moved on.
	 */
	crumb->setChecked(is_current);
	if (is_ahead) {
		if (crumb->graphicsEffect() == nullptr) {
			auto* faded = new QGraphicsOpacityEffect(crumb);
			faded->setOpacity(0.4);
			crumb->setGraphicsEffect(faded);
		}
	} else {
		crumb->setGraphicsEffect(nullptr);
	}
}
