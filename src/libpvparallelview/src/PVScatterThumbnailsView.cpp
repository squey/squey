//
// MIT License
//
// © Squey, 2026
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

#include <pvparallelview/PVScatterThumbnailsView.h>

#include <pvparallelview/PVDisplayViewScatter.h>
#include <pvparallelview/PVScatterThumbnail.h>
#include <pvparallelview/PVScatterThumbnailsModel.h>

#include <pvdisplays/PVDisplayIf.h>
#include <pvdisplays/PVDisplaysContainer.h>

#include <pvkernel/core/PVConfig.h>
#include <pvkernel/core/qobject_helpers.h>

#include <squey/PVView.h>

#include <QApplication>
#include <QKeyEvent>
#include <QClipboard>
#include <QComboBox>
#include <QCursor>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListView>
#include <QMenu>
#include <QPainter>
#include <QScrollBar>
#include <QSlider>
#include <QSortFilterProxyModel>
#include <QStyle>
#include <QStyleOptionComboBox>
#include <QTextLayout>
#include <QTimer>
#include <QVBoxLayout>

namespace
{

/**
 * How long the pointer must sit on a thumbnail before the enlarged preview
 * appears.
 *
 * The preview is rendered synchronously and at a higher resolution than the
 * thumbnail, so it must not be triggered by a pointer merely crossing the
 * gallery.
 */
constexpr int preview_delay_ms = 600;

//! Room left under a thumbnail for its label; fixed, so rows stay aligned.
constexpr int label_lines = 2;

//! Padding between a cell's edge, its thumbnail and its label.
constexpr int cell_margin = 4;

/**
 * Quiet time after the last size-slider move before the gallery re-renders.
 *
 * Long enough that dragging the slider across its range costs one re-render
 * rather than one per step.
 */
constexpr int size_settle_ms = 250;

/**
 * Where the toolbar state is remembered between two openings.
 *
 * Thumbnail size and sort order are how the user has chosen to read the
 * gallery, not something about one dataset, so they belong in the application
 * configuration. The axis filter is deliberately left out: reopening on a
 * gallery silently showing a fraction of its pairs would read as a bug.
 */
constexpr const char* size_setting = "scatter_thumbnails/thumbnail_size";
constexpr const char* sort_setting = "scatter_thumbnails/sort_by_correlation";

/**
 * Widen @a combo so the theme's drop-down arrow cannot eat into its text.
 *
 * The application stylesheet draws a 12 px arrow with
 * "subcontrol-origin: padding" over a 4 px right padding (see QComboBox in
 * theme-dark.qss, where that padding was cut down from 36). AdjustToContents
 * only reserves the padding the stylesheet declares, so what is actually left
 * for the arrow is a couple of pixels: the text fits under some fonts and is
 * elided under others. Reserving the arrow width the style reports makes it fit
 * whatever the font.
 */
void reserve_room_for_arrow(QComboBox* combo)
{
	QStyleOptionComboBox option;
	option.initFrom(combo);
	option.rect = QRect(QPoint(0, 0), combo->sizeHint());

	const int arrow_width =
	    combo->style()
	        ->subControlRect(QStyle::CC_ComboBox, &option, QStyle::SC_ComboBoxArrow, combo)
	        .width();

	combo->setMinimumWidth(combo->sizeHint().width() + arrow_width);
}

} // namespace

int PVParallelView::PVScatterThumbnailDelegate::label_height(QFontMetrics const& metrics) const
{
	return metrics.lineSpacing() * label_lines;
}

QSize PVParallelView::PVScatterThumbnailDelegate::sizeHint(QStyleOptionViewItem const& option,
                                                           QModelIndex const&) const
{
	// Independent of the item: every cell is the same size, whatever the pixmap
	// it currently holds and however long its axis names are.
	const int height = _display_size + label_height(option.fontMetrics) + 3 * cell_margin;
	return QSize(_display_size + 2 * cell_margin, height);
}

void PVParallelView::PVScatterThumbnailDelegate::paint(QPainter* painter,
                                                       QStyleOptionViewItem const& option,
                                                       QModelIndex const& index) const
{
	painter->save();

	const bool selected = option.state & QStyle::State_Selected;
	if (selected) {
		painter->fillRect(option.rect, option.palette.highlight());
	}

	// Centred in the cell: the view may hand out a rect wider than the size
	// hint asked for when it spreads the columns.
	const QRect thumbnail_rect(option.rect.left() + (option.rect.width() - _display_size) / 2,
	                           option.rect.top() + cell_margin, _display_size, _display_size);

	const QPixmap pixmap = index.data(Qt::DecorationRole).value<QPixmap>();
	if (not pixmap.isNull()) {
		// Scaled to the cell rather than drawn at its own size: while the size
		// slider moves, the cached pixmaps still carry the size they were
		// rendered at, and the sharp re-render only follows once it settles.
		painter->setRenderHint(QPainter::SmoothPixmapTransform);
		painter->drawPixmap(thumbnail_rect, pixmap);
	}

	const QRect text_rect(option.rect.left() + cell_margin, thumbnail_rect.bottom() + cell_margin,
	                      option.rect.width() - 2 * cell_margin, label_height(option.fontMetrics));

	painter->setPen(selected ? option.palette.highlightedText().color()
	                         : option.palette.text().color());
	painter->setClipRect(text_rect);

	// Wrapped over at most label_lines lines with the last one elided. Laid out
	// by hand: drawText() with TextWordWrap would spill past the cell, and
	// elidedText() alone cannot wrap.
	const QString text = index.data(Qt::DisplayRole).toString();
	QTextLayout layout(text, painter->font());
	QTextOption text_option;
	text_option.setAlignment(Qt::AlignHCenter);
	text_option.setWrapMode(QTextOption::WrapAtWordBoundaryOrAnywhere);
	layout.setTextOption(text_option);

	QString elided;
	layout.beginLayout();
	qreal y = 0;
	for (int i = 0; i < label_lines; ++i) {
		QTextLine line = layout.createLine();
		if (not line.isValid()) {
			break;
		}
		line.setLineWidth(text_rect.width());
		line.setPosition(QPointF(0, y));
		y += line.height();

		// Text left over once the last allowed line is filled: that line is
		// replaced by an elided one, drawn below instead of by the layout.
		if (i == label_lines - 1 and line.textStart() + line.textLength() < text.length()) {
			elided = option.fontMetrics.elidedText(text.mid(line.textStart()), Qt::ElideRight,
			                                       text_rect.width());
		}
	}
	layout.endLayout();

	const QPointF origin(text_rect.left(), text_rect.top());
	for (int i = 0; i < layout.lineCount(); ++i) {
		QTextLine line = layout.lineAt(i);
		if (not elided.isEmpty() and i == layout.lineCount() - 1) {
			painter->drawText(QRectF(text_rect.left(), text_rect.top() + line.y(),
			                         text_rect.width(), line.height()),
			                  Qt::AlignHCenter | Qt::AlignTop, elided);
		} else {
			line.draw(painter, origin);
		}
	}

	painter->restore();
}

PVParallelView::PVScatterThumbnailsView::PVScatterThumbnailsView(Squey::PVView& view,
                                                                 QWidget* parent)
    : QWidget(parent)
    , _view(view)
    , _mouse_buttons_legend("Open scatter view (double click)", "Context menu", "Scroll")
    , _help_widget(this)
{
	auto* layout = new QVBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);

	setup_toolbar(layout);

	_model = new PVScatterThumbnailsModel(_view, this);

	_proxy = new QSortFilterProxyModel(this);
	_proxy->setSourceModel(_model);
	// Sorted on demand only: every thumbnail landing emits dataChanged, and a
	// dynamic sort would reshuffle the gallery under the user's pointer each
	// time one did.
	_proxy->setDynamicSortFilter(false);

	_list = new QListView(this);
	_list->setModel(_proxy);
	_list->setViewMode(QListView::IconMode);
	_list->setResizeMode(QListView::Adjust);
	_list->setMovement(QListView::Static);
	_list->setUniformItemSizes(true);
	_list->setSelectionMode(QAbstractItemView::SingleSelection);
	_list->setSpacing(2);

	// The delegate owns the cell geometry, so no grid size is set here: an
	// explicit one would override its sizeHint and reintroduce the mismatch
	// between cell and content.
	_delegate = new PVScatterThumbnailDelegate(this);
	_list->setItemDelegate(_delegate);
	// entered() is emitted from the viewport's move events, which only reach
	// the view with a button held down unless the viewport itself tracks.
	_list->viewport()->setMouseTracking(true);
	_list->setContextMenuPolicy(Qt::CustomContextMenu);
	layout->addWidget(_list);

	connect(_list, &QAbstractItemView::doubleClicked, this,
	        &PVScatterThumbnailsView::open_scatter_view);
	connect(_list, &QWidget::customContextMenuRequested, this,
	        &PVScatterThumbnailsView::show_context_menu);
	connect(_list, &QAbstractItemView::entered, this, &PVScatterThumbnailsView::schedule_preview);
	connect(_list, &QAbstractItemView::viewportEntered, this,
	        &PVScatterThumbnailsView::hide_preview);
	// Scrolling moves the thumbnails out from under a preview that is still
	// showing the one that used to be there.
	connect(_list->verticalScrollBar(), &QScrollBar::valueChanged, this,
	        &PVScatterThumbnailsView::hide_preview);

	_preview_timer = new QTimer(this);
	_preview_timer->setSingleShot(true);
	_preview_timer->setInterval(preview_delay_ms);
	connect(_preview_timer, &QTimer::timeout, this, &PVScatterThumbnailsView::show_preview);

	connect(_model, &PVScatterThumbnailsModel::correlations_ready, this, [this]() {
		if (_sort_combo->currentIndex() == 1) {
			apply_sorting(true);
		}
	});

	// A repaint is what asks for thumbnails, and it only asks for the items it
	// paints -- so a screenful bigger than the model's request queue leaves the
	// rest unasked, showing an empty cell until something else repaints it
	// (moving the pointer over it, say). Repainting once the queue drains asks
	// for what is left; it settles as soon as everything visible is cached.
	connect(_model, &PVScatterThumbnailsModel::renders_idle, this,
	        [this]() { _list->viewport()->update(); });

	connect(_model, &PVScatterThumbnailsModel::preview_ready, this,
	        &PVScatterThumbnailsView::display_preview);

	apply_grid_size(_size_slider->value());
	update_counters();

	// Correlation first: the gallery exists to point at the pairs worth
	// opening, and axes order says nothing about which those are. Setting the
	// combo is enough -- its currentIndexChanged does the sorting.
	const QVariant stored_sort = PVCore::PVConfig::value(sort_setting);
	const bool by_correlation = not stored_sort.isValid() or stored_sort.toBool();
	_sort_combo->setCurrentIndex(by_correlation ? 1 : 0);
	if (not by_correlation) {
		// setCurrentIndex(0) on a combo already at 0 emits nothing.
		apply_sorting(false);
	}

	_help_widget.hide();
	_help_widget.initTextFromFile("scatter thumbnails' help");
	_help_widget.addTextFromFile(":help-selection");
	_help_widget.addTextFromFile(":help-layers");
	_help_widget.newColumn();
	_help_widget.addTextFromFile(":help-lines");
	_help_widget.addTextFromFile(":help-application");
	_help_widget.newTable();
	_help_widget.addTextFromFile(":help-mouse-scatter-thumbnails-view");
	_help_widget.newColumn();
	_help_widget.addTextFromFile(":help-shortcuts-scatter-thumbnails-view");
	_help_widget.finalizeText();

	// Autonomous view: it borrows nothing from a PVViewRenderingContext, so it
	// is tied straight to the model, as PVSeriesViewWidget is. Deleting the
	// widget here drains the renders still reading the model's columns.
	_view_deleted_connection = _view._about_to_be_delete.connect([this] { delete this; });
}

PVParallelView::PVScatterThumbnailsView::~PVScatterThumbnailsView()
{
	// Renders first, before anything else starts going away: they read the
	// model view's scaled columns and its colour buffer from a worker thread,
	// and every line below tears down something around them. Waiting for the
	// model's own destructor to do it leaves them running through the whole
	// teardown, because QObject only deletes its children once the members
	// above have gone.
	_model->drain();

	// Then the chain that reads the model. QObject deletes children in
	// creation order and the model was created before the proxy and the list,
	// so it would otherwise go first and leave those two reacting to its
	// destruction -- a reset, and the repaint that follows -- in the middle of
	// a widget that is already half gone.
	_list->setModel(nullptr);
	_proxy->setSourceModel(nullptr);

	// Free-standing window, so it is not taken down by the widget tree.
	delete _preview;
}

void PVParallelView::PVScatterThumbnailsView::setup_toolbar(QBoxLayout* layout)
{
	auto* bar = new QHBoxLayout;
	bar->setContentsMargins(6, 4, 6, 4);

	_size_slider = new QSlider(Qt::Horizontal, this);
	_size_slider->setRange(PVScatterThumbnailsModel::min_size, PVScatterThumbnailsModel::max_size);
	// PVConfig::value() returns an invalid variant for a key never written, so
	// the fallback is applied here rather than passed in.
	const QVariant stored_size = PVCore::PVConfig::value(size_setting);
	_size_slider->setValue(stored_size.isValid() ? stored_size.toInt()
	                                             : PVScatterThumbnailsModel::default_size);
	_size_slider->setFixedWidth(120);
	_size_slider->setToolTip(tr("Thumbnail size"));

	_size_apply_timer = new QTimer(this);
	_size_apply_timer->setSingleShot(true);
	_size_apply_timer->setInterval(size_settle_ms);
	connect(_size_apply_timer, &QTimer::timeout, this,
	        &PVScatterThumbnailsView::apply_render_size);

	connect(_size_slider, &QSlider::valueChanged, this, [this](int size) {
		apply_grid_size(size);
		_size_apply_timer->start();
		PVCore::PVConfig::set_value(size_setting, size);
	});

	_sort_combo = new QComboBox(this);
	_sort_combo->addItem(tr("Axes order"));
	_sort_combo->addItem(tr("Correlation"));
	// Otherwise the combo takes the width of its first item and elides the rest.
	_sort_combo->setSizeAdjustPolicy(QComboBox::AdjustToContents);
	reserve_room_for_arrow(_sort_combo);
	connect(_sort_combo, &QComboBox::currentIndexChanged, this, [this](int index) {
		PVCore::PVConfig::set_value(sort_setting, index == 1);
		apply_sorting(index == 1);
	});

	// Ranking answers "which pairs are worth a look"; this answers "what does
	// this one axis look like against the others", which no amount of sorting
	// does and which is the only way through a gallery of several thousand
	// pairs.
	_filter_edit = new QLineEdit(this);
	_filter_edit->setPlaceholderText(tr("Filter by axis name"));
	_filter_edit->setClearButtonEnabled(true);
	_filter_edit->setFixedWidth(160);
	connect(_filter_edit, &QLineEdit::textChanged, this, &PVScatterThumbnailsView::apply_filter);

	_counter_label = new QLabel(this);

	bar->addWidget(new QLabel(tr("Size:"), this));
	bar->addWidget(_size_slider);
	bar->addSpacing(12);
	bar->addWidget(new QLabel(tr("Sort by:"), this));
	bar->addWidget(_sort_combo);
	bar->addSpacing(12);
	bar->addWidget(_filter_edit);
	bar->addStretch();
	bar->addWidget(_counter_label);

	layout->addLayout(bar);
}

int PVParallelView::PVScatterThumbnailsView::source_row(QModelIndex const& index) const
{
	return index.isValid() ? _proxy->mapToSource(index).row() : -1;
}

void PVParallelView::PVScatterThumbnailsView::open_scatter_view(QModelIndex const& index)
{
	const int row = source_row(index);
	if (row < 0) {
		return;
	}

	auto const [x_col, y_col] = _model->axes_at(row);
	if (x_col == PVCol() or y_col == PVCol()) {
		return;
	}

	// The gallery is docked in the workspace, which is the container; asking
	// it to build the widget is what the axis menu already does to open a
	// scatter view (PVDisplayViewScatter::add_to_axis_menu).
	auto* container = PVCore::get_qobject_parent_of_type<PVDisplays::PVDisplaysContainer*>(this);
	if (container == nullptr) {
		return;
	}

	hide_preview();
	container->create_view_widget(
	    PVDisplays::display_view_if<PVDisplays::PVDisplayViewScatter>(), &_view, {x_col, y_col});
}

void PVParallelView::PVScatterThumbnailsView::show_context_menu(QPoint const& pos)
{
	const QModelIndex index = _list->indexAt(pos);
	const int row = source_row(index);
	if (row < 0) {
		return;
	}

	hide_preview();

	QMenu menu(this);
	QAction* open = menu.addAction(tr("Open as scatter view"));
	QAction* swap = menu.addAction(tr("Swap X/Y axes"));
	menu.addSeparator();
	QAction* copy = menu.addAction(tr("Copy image"));

	QAction* chosen = menu.exec(QCursor::pos());
	if (chosen == open) {
		open_scatter_view(index);
	} else if (chosen == swap) {
		_model->swap_axes(row);
	} else if (chosen == copy) {
		QApplication::clipboard()->setPixmap(
		    _model->render_preview(row, PVScatterThumbnailsModel::preview_size));
	}
}

void PVParallelView::PVScatterThumbnailsView::schedule_preview(QModelIndex const& index)
{
	if (index == _preview_index and _preview != nullptr and _preview->isVisible()) {
		return;
	}

	hide_preview();
	_preview_index = index;
	_preview_timer->start();
}

void PVParallelView::PVScatterThumbnailsView::show_preview()
{
	const int row = source_row(_preview_index);
	if (row < 0) {
		return;
	}

	// Asked for, not rendered here: a preview is several times the area of a
	// thumbnail, and rendering it on this thread would freeze the application
	// under the pointer on a large file. It arrives through preview_ready().
	_model->request_preview(row, PVScatterThumbnailsModel::preview_size);
}

void PVParallelView::PVScatterThumbnailsView::display_preview(int row, QPixmap const& pixmap)
{
	// The pointer may have left the item, or moved to another one, while this
	// was rendering.
	if (pixmap.isNull() or row != source_row(_preview_index)) {
		return;
	}

	if (_preview == nullptr) {
		// Its own window rather than a child: a thumbnail's enlarged render is
		// several times the size of the item it sits on, so it has to be free
		// to spill outside the gallery.
		_preview = new QLabel(nullptr, Qt::ToolTip | Qt::FramelessWindowHint);
		_preview->setAttribute(Qt::WA_ShowWithoutActivating);
	}

	_preview->setPixmap(pixmap);
	_preview->setToolTip(_model->label_at(row));
	_preview->adjustSize();
	_preview->move(QCursor::pos() + QPoint(16, 16));
	_preview->show();
}

void PVParallelView::PVScatterThumbnailsView::hide_preview()
{
	_preview_timer->stop();
	if (_preview != nullptr) {
		_preview->hide();
	}
	_preview_index = QModelIndex();
}

void PVParallelView::PVScatterThumbnailsView::apply_grid_size(int size)
{
	_delegate->set_display_size(size);
	// uniformItemSizes caches the first item's sizeHint, so the new one is only
	// picked up once the layout is redone.
	_list->doItemsLayout();
}

void PVParallelView::PVScatterThumbnailsView::apply_render_size()
{
	_model->set_thumbnail_size(_size_slider->value());
}

void PVParallelView::PVScatterThumbnailsView::apply_sorting(bool by_correlation)
{
	if (not by_correlation) {
		_proxy->sort(-1);
		update_counters();
		return;
	}

	if (not _model->has_correlations()) {
		// Scoring every pair takes a moment; sorting on a half-filled set would
		// keep re-ordering, so the gallery waits for correlations_ready().
		_counter_label->setText(tr("Ranking pairs…"));
		_model->compute_correlations();
		return;
	}

	_proxy->setSortRole(PVScatterThumbnailsModel::CorrelationRole);
	_proxy->sort(0, Qt::DescendingOrder);
	update_counters();
}

void PVParallelView::PVScatterThumbnailsView::apply_filter(QString const& text)
{
	// Matches against the "x \u00d7 y" label, so typing an axis name keeps every
	// pair it takes part in, on either side.
	_proxy->setFilterCaseSensitivity(Qt::CaseInsensitive);
	_proxy->setFilterFixedString(text);

	// The proxy is not dynamic, so re-filtering drops the arrangement and the
	// sort has to be asked for again.
	apply_sorting(_sort_combo->currentIndex() == 1);
}

void PVParallelView::PVScatterThumbnailsView::update_counters()
{
	const int shown = _proxy->rowCount();
	const int total = _model->rowCount();

	QString text = shown == total ? tr("%n pair(s)", "", total)
	                              : tr("%1 of %2 pairs").arg(shown).arg(total);
	if (_view.get_row_count() > PVScatterThumbnail::row_sampling_threshold) {
		text += tr(" — sampled");
	}
	_counter_label->setText(text);
}

void PVParallelView::PVScatterThumbnailsView::changeEvent(QEvent* event)
{
	QWidget::changeEvent(event);

	// Switching theme reloads the stylesheet, and the arrow allowance was
	// measured against the old one.
	if (_sort_combo != nullptr and
	    (event->type() == QEvent::StyleChange or event->type() == QEvent::FontChange)) {
		reserve_room_for_arrow(_sort_combo);
	}
}

void PVParallelView::PVScatterThumbnailsView::keyPressEvent(QKeyEvent* event)
{
	if (PVWidgets::PVHelpWidget::is_help_key(event->key())) {
		if (_help_widget.isHidden()) {
			_help_widget.popup(this, PVWidgets::PVTextPopupWidget::AlignTop,
			                   PVWidgets::PVTextPopupWidget::ExpandAll);
		}
		return;
	}

	QWidget::keyPressEvent(event);
}

void PVParallelView::PVScatterThumbnailsView::enterEvent(QEnterEvent* event)
{
	Q_EMIT set_status_bar_mouse_legend(_mouse_buttons_legend);
	QWidget::enterEvent(event);
}

void PVParallelView::PVScatterThumbnailsView::leaveEvent(QEvent* event)
{
	hide_preview();
	Q_EMIT clear_status_bar_mouse_legend();
	QWidget::leaveEvent(event);
}
