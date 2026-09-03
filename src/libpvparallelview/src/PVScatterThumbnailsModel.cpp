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

#include <pvparallelview/PVScatterThumbnailsModel.h>

#include <pvparallelview/common.h>

#include <squey/PVScaled.h>
#include <squey/PVView.h>

#include <pvhwloc.h>

#include <tbb/parallel_for.h>

#include <QPainter>

#include <algorithm>
#include <unordered_set>

namespace
{

/**
 * Pending requests kept before the oldest ones are dropped.
 *
 * A fast scroll asks for far more thumbnails than it ever displays. Requests
 * are served newest first, so the ones piling up at the front are exactly the
 * ones the user has already scrolled past.
 *
 * Sized to hold a whole screenful at the smallest thumbnail size, so that one
 * repaint is not truncated; anything the bound does drop is picked up by the
 * repaint renders_idle() triggers.
 */
constexpr size_t max_pending_requests = 512;

/**
 * Opacity the background layer is drawn at, under the selection layer.
 *
 * Same value as PVScatterView::drawBackground, so a thumbnail and the real
 * scatter view read the same way.
 */
constexpr qreal background_opacity = 0.25;

} // namespace

PVParallelView::PVScatterThumbnailsModel::PVScatterThumbnailsModel(Squey::PVView& view,
                                                                   QObject* parent)
    : QAbstractListModel(parent)
    , _view(view)
    , _cancelled(std::make_shared<std::atomic<bool>>(false))
    , _correlations_cancelled(std::make_shared<std::atomic<bool>>(false))
    , _preview_cancelled(std::make_shared<std::atomic<bool>>(false))
    , _max_concurrent_renders(std::max<size_t>(1, pvhwloc::core_count()))
    , _max_cached_bytes(64u << 20)
{
	rebuild_placeholder();
	reset_pairs();

	// Subscribed straight to the model: this view holds no zone and no
	// processor, so it has nothing to learn from a PVViewRenderingContext (see
	// the connection rule in PVViewRenderingContext.h).
	_selection_changed_connection = _view._selection_view_changed.connect(
	    sigc::mem_fun(*this, &PVScatterThumbnailsModel::on_selection_changed));
	_output_selection_connection = _view._update_output_selection.connect(
	    sigc::mem_fun(*this, &PVScatterThumbnailsModel::on_selection_changed));
	_output_layer_connection = _view._update_output_layer.connect(
	    sigc::mem_fun(*this, &PVScatterThumbnailsModel::on_output_layer_changed));

	// A scaling pass rewrites the very columns a render walks. It recomputes
	// them in place (see PVScaled::create_table), so an in-flight render reads
	// mixed values rather than freed memory, but the images it produces are
	// stale either way.
	_scaling_connection = _view.get_parent<Squey::PVScaled>()._scaled_updated.connect(
	    [this](QList<PVCol> const&) { on_scaling_changed(); });

	// Not routed through the rendering context: this is a display setting, not
	// zone state, so it comes straight from the model (see the connection rule
	// in PVViewRenderingContext.h). PVScatterView reads it the same way.
	_unselected_zombie_connection = _view._toggle_unselected_zombie_visibility.connect(
	    sigc::mem_fun(*this, &PVScatterThumbnailsModel::on_unselected_zombie_visibility_toggled));

	_axes_comb_about_to_change_connection = _view._axis_combination_about_to_update.connect(
	    sigc::mem_fun(*this, &PVScatterThumbnailsModel::on_axes_combination_about_to_change));
	_axes_comb_changed_connection =
	    _view._axis_combination_updated.connect([this](bool) { on_axes_combination_changed(); });
}

PVParallelView::PVScatterThumbnailsModel::~PVScatterThumbnailsModel() noexcept
{
	// Every render dereferences the scaled columns and the colour buffer of the
	// model view; none may still be running once this returns.
	drain();
}

void PVParallelView::PVScatterThumbnailsModel::cancel_renders()
{
	_cancelled->store(true);

	// The completion callbacks of the tasks just cancelled are still queued on
	// this thread. They test this flag, so they will drop their images; and
	// they are what brings _running back down, which is why it is not reset
	// here.
	_cancelled = std::make_shared<std::atomic<bool>>(false);
	_pending.clear();

	// Nothing is running any more, so no row is in flight. Clearing this now
	// rather than leaving it to the queued callbacks matters: a repaint landing
	// in between would otherwise refuse to re-request those rows and leave them
	// showing a placeholder until the next repaint. A render still running at
	// this point will drop its result on the cancellation flag, and its row is
	// re-requested by the repaint that follows.
	_in_flight.clear();
}

void PVParallelView::PVScatterThumbnailsModel::request_preview(int row, int size)
{
	if (_shutting_down or row < 0 or row >= int(_pairs.size())) {
		return;
	}

	// The pointer has moved: whatever was being rendered for the previous item
	// is of no use now.
	_preview_cancelled->store(true);
	_preview_cancelled = std::make_shared<std::atomic<bool>>(false);

	const Pair pair = _pairs[row];
	auto cancelled = _preview_cancelled;

	_tasks.run([this, row, pair, size, cancelled]() {
		PVScatterThumbnailImages images;
		if (not PVScatterThumbnail::render(_view, pair.x, pair.y, size, images, *cancelled)) {
			return;
		}

		// Composed on the UI thread: a QPixmap cannot be built anywhere else.
		QMetaObject::invokeMethod(
		    this,
		    [this, row, images = std::move(images), cancelled]() mutable {
			    if (cancelled->load()) {
				    return;
			    }
			    Q_EMIT preview_ready(row, compose(images));
			},
		    Qt::QueuedConnection);
	});
}

void PVParallelView::PVScatterThumbnailsModel::drain()
{
	cancel_renders();
	_preview_cancelled->store(true);
	// The correlation pass reads the same columns, and has its own flag so that
	// a selection change does not stop it.
	cancel_correlations();
	// Only here, where a caller is about to take away what those two read.
	_tasks.wait();
}

void PVParallelView::PVScatterThumbnailsModel::cancel_correlations()
{
	_correlations_cancelled->store(true);
	_correlations_cancelled = std::make_shared<std::atomic<bool>>(false);
	_correlations_pending = false;
}

void PVParallelView::PVScatterThumbnailsModel::detach()
{
	_shutting_down = true;
	drain();

	// Dropped by hand rather than by the PVDisconnector members: those run
	// from the destructor, which is well after the widget has begun to
	// disappear around them.
	_selection_changed_connection.disconnect();
	_output_selection_connection.disconnect();
	_output_layer_connection.disconnect();
	_scaling_connection.disconnect();
	_unselected_zombie_connection.disconnect();
	_axes_comb_about_to_change_connection.disconnect();
	_axes_comb_changed_connection.disconnect();
}

void PVParallelView::PVScatterThumbnailsModel::reset_pairs()
{
	beginResetModel();

	_pairs.clear();
	_correlations.clear();
	cancel_correlations();

	// Pairs are taken from the current axes combination, in its current order.
	// A combination may list the same column twice; the same pair is only worth
	// showing once, and a column against itself not at all.
	std::vector<PVCol> const& comb = _view.get_axes_combination().get_combination();
	std::unordered_set<uint64_t> seen;
	for (size_t i = 0; i < comb.size(); i++) {
		for (size_t j = i + 1; j < comb.size(); j++) {
			if (comb[i] == comb[j]) {
				continue;
			}
			const uint64_t key =
			    (uint64_t(uint32_t(comb[i].value())) << 32) | uint32_t(comb[j].value());
			if (seen.insert(key).second) {
				_pairs.push_back({comb[i], comb[j]});
			}
		}
	}

	endResetModel();
}

int PVParallelView::PVScatterThumbnailsModel::rowCount(QModelIndex const& parent) const
{
	return parent.isValid() ? 0 : int(_pairs.size());
}

std::pair<PVCol, PVCol> PVParallelView::PVScatterThumbnailsModel::axes_at(int row) const
{
	if (row < 0 or row >= int(_pairs.size())) {
		return {PVCol(), PVCol()};
	}
	return {_pairs[row].x, _pairs[row].y};
}

QString PVParallelView::PVScatterThumbnailsModel::label_at(int row) const
{
	if (row < 0 or row >= int(_pairs.size())) {
		return {};
	}
	return QStringLiteral("%1 × %2")
	    .arg(_view.get_nraw_axis_name(_pairs[row].x), _view.get_nraw_axis_name(_pairs[row].y));
}

QVariant PVParallelView::PVScatterThumbnailsModel::data(QModelIndex const& index, int role) const
{
	const int row = index.row();
	if (not index.isValid() or row >= int(_pairs.size())) {
		return {};
	}

	switch (role) {
	case Qt::DisplayRole:
		return label_at(row);

	case Qt::DecorationRole: {
		if (auto it = _cache.find(row); it != _cache.end()) {
			touch_cache(row);
			return it->second.pixmap;
		}
		// Only the items a view is about to paint are queried, so asking here
		// is what keeps the gallery to the renders the user can see.
		request_render(row);
		return _placeholder;
	}

	case Qt::ToolTipRole: {
		QString tip = label_at(row);
		if (row < int(_correlations.size())) {
			tip += QStringLiteral("\ncorrelation: %1").arg(_correlations[row], 0, 'f', 3);
		}
		if (auto it = _cache.find(row); it != _cache.end() and it->second.sampled) {
			tip += QStringLiteral("\n(rendered from a sample of the rows)");
		}
		return tip;
	}

	case CorrelationRole:
		return row < int(_correlations.size()) ? QVariant(_correlations[row]) : QVariant(0.);

	case SampledRole: {
		auto it = _cache.find(row);
		return it != _cache.end() and it->second.sampled;
	}

	default:
		return {};
	}
}

void PVParallelView::PVScatterThumbnailsModel::request_render(int row) const
{
	// Between the two axes-combination signals the pair list is about to be
	// rebuilt, so anything rendered against it would be thrown away at once.
	if (_shutting_down or _axes_combination_changing or _in_flight.contains(row)) {
		return;
	}

	if (std::find(_pending.begin(), _pending.end(), row) != _pending.end()) {
		return;
	}

	_pending.push_back(row);
	// Served newest first, so the front holds the requests scrolled past.
	while (_pending.size() > max_pending_requests) {
		_pending.pop_front();
	}

	schedule_renders();
}

void PVParallelView::PVScatterThumbnailsModel::schedule_renders() const
{
	auto* self = const_cast<PVScatterThumbnailsModel*>(this);

	while (not _shutting_down and _running < _max_concurrent_renders and not _pending.empty()) {
		const int row = _pending.back();
		_pending.pop_back();

		if (_cache.contains(row) or _in_flight.contains(row)) {
			continue;
		}

		const Pair pair = _pairs[row];
		const int size = _size;
		auto cancelled = _cancelled;

		++_running;
		_in_flight.insert(row);
		_tasks.run([this, self, row, pair, size, cancelled]() {
			PVScatterThumbnailImages images;
			const bool rendered =
			    PVScatterThumbnail::render(_view, pair.x, pair.y, size, images, *cancelled);

			// Queued to the UI thread: the pixmap must be built there, and the
			// cache is only ever touched from there. Qt drops a queued call
			// whose receiver died first, so this cannot outlive the model.
			QMetaObject::invokeMethod(
			    self,
			    [self, row, pair, images = std::move(images), size, rendered,
			     cancelled]() mutable {
				    --self->_running;
				    self->_in_flight.erase(row);
				    // A thumbnail rendered against data, a size or an axis order
				    // that has since changed would be shown as if it were
				    // current -- the last of which is what swap_axes does to a
				    // row whose render is already in flight.
				    if (rendered and not cancelled->load() and size == self->_size and
				        row < int(self->_pairs.size()) and self->_pairs[row].x == pair.x and
				        self->_pairs[row].y == pair.y) {
					    self->store_render(row, images);
				    }
				    self->schedule_renders();

				    // Nothing left to render: let the gallery ask again for
				    // whatever is on screen without an image, which the bound
				    // on the queue may have dropped.
				    if (self->_running == 0 and self->_pending.empty() and
				        self->_stored_since_idle) {
					    self->_stored_since_idle = false;
					    Q_EMIT self->renders_idle();
				    }
			    },
			    Qt::QueuedConnection);
		});
	}
}

void PVParallelView::PVScatterThumbnailsModel::store_render(int row,
                                                            PVScatterThumbnailImages const& images)
{
	if (row >= int(_pairs.size())) {
		return;
	}

	// A render can land on a row that already has an image (a swap_axes then a
	// re-request); its LRU entry must go, or the list would hold the row twice.
	if (auto it = _cache.find(row); it != _cache.end()) {
		_lru.erase(it->second.lru);
		_cached_bytes -= it->second.bytes;
	}

	QPixmap pixmap = compose(images);
	// 4 bytes per pixel, which is what the ARGB pixmap the gallery draws costs.
	const size_t bytes = size_t(pixmap.width()) * size_t(pixmap.height()) * 4;

	_lru.push_front(row);
	_cache[row] = CacheEntry{std::move(pixmap), images.sampled, _lru.begin(), bytes};
	_cached_bytes += bytes;
	_stored_since_idle = true;
	trim_cache();

	const QModelIndex idx = index(row);
	Q_EMIT dataChanged(idx, idx, {Qt::DecorationRole, Qt::ToolTipRole, SampledRole});
}

QPixmap PVParallelView::PVScatterThumbnailsModel::compose(
    PVScatterThumbnailImages const& images) const
{
	QPixmap pixmap(images.bg.size());
	pixmap.fill(color_view_bg);

	QPainter painter(&pixmap);

	// Background dimmed under a full-opacity selection, as
	// PVScatterView::drawBackground composes its own two layers -- including
	// dropping the background entirely when the user has hidden the unselected
	// and zombie lines, which is a view-wide setting the gallery must follow or
	// it shows something else than the views beside it.
	if (_view.are_view_unselected_zombie_visible()) {
		// With nothing selected the render skips the selection layer, and the
		// background is then all there is to show, at full opacity.
		painter.setOpacity(images.sel.isNull() ? 1. : background_opacity);
		painter.drawImage(0, 0, images.bg);
	}

	if (not images.sel.isNull()) {
		painter.setOpacity(1.);
		painter.drawImage(0, 0, images.sel);
	}

	return pixmap;
}

void PVParallelView::PVScatterThumbnailsModel::rebuild_placeholder()
{
	_placeholder = QPixmap(_size, _size);
	_placeholder.fill(color_view_bg);
}

void PVParallelView::PVScatterThumbnailsModel::touch_cache(int row) const
{
	auto it = _cache.find(row);
	if (it == _cache.end()) {
		return;
	}
	_lru.erase(it->second.lru);
	_lru.push_front(row);
	it->second.lru = _lru.begin();
}

void PVParallelView::PVScatterThumbnailsModel::trim_cache() const
{
	// One thumbnail always stays, whatever the ceiling: dropping the one just
	// rendered would loop forever on a size the ceiling cannot hold.
	while (_cached_bytes > _max_cached_bytes and _lru.size() > 1) {
		auto it = _cache.find(_lru.back());
		if (it != _cache.end()) {
			_cached_bytes -= it->second.bytes;
			_cache.erase(it);
		}
		_lru.pop_back();
	}
}

void PVParallelView::PVScatterThumbnailsModel::invalidate_thumbnails()
{
	cancel_renders();

	_cache.clear();
	_lru.clear();
	_cached_bytes = 0;

	if (not _pairs.empty()) {
		Q_EMIT dataChanged(index(0), index(int(_pairs.size()) - 1),
		                   {Qt::DecorationRole, Qt::ToolTipRole, SampledRole});
	}
}

void PVParallelView::PVScatterThumbnailsModel::set_thumbnail_size(int size)
{
	if (size == _size) {
		return;
	}
	_size = size;
	rebuild_placeholder();
	invalidate_thumbnails();
}

void PVParallelView::PVScatterThumbnailsModel::swap_axes(int row)
{
	if (row < 0 or row >= int(_pairs.size())) {
		return;
	}

	std::swap(_pairs[row].x, _pairs[row].y);

	// Only this pair changed, so only its thumbnail is dropped; the renders of
	// the others stay valid and must not be thrown away. Its correlation is
	// left alone: the coefficient is symmetric.
	if (auto it = _cache.find(row); it != _cache.end()) {
		_lru.erase(it->second.lru);
		_cached_bytes -= it->second.bytes;
		_cache.erase(it);
	}

	const QModelIndex idx = index(row);
	Q_EMIT dataChanged(idx, idx, {Qt::DisplayRole, Qt::DecorationRole, Qt::ToolTipRole});
}

QPixmap PVParallelView::PVScatterThumbnailsModel::render_preview(int row, int size) const
{
	if (row < 0 or row >= int(_pairs.size())) {
		return {};
	}

	const std::atomic<bool> not_cancelled{false};
	PVScatterThumbnailImages images;
	if (not PVScatterThumbnail::render(_view, _pairs[row].x, _pairs[row].y, size, images,
	                                   not_cancelled)) {
		return {};
	}

	return compose(images);
}

void PVParallelView::PVScatterThumbnailsModel::compute_correlations()
{
	if (_shutting_down or _pairs.empty() or not _correlations.empty() or _correlations_pending) {
		return;
	}

	_correlations_pending = true;
	auto cancelled = _correlations_cancelled;
	auto scores = std::make_shared<std::vector<double>>(_pairs.size(), 0.);

	_tasks.run([this, scores, cancelled]() {
		// The per-column sums first, once per column: with N axes each column
		// takes part in N-1 pairs, so folding them into the pair loop would walk
		// every column N-1 times over.
		std::vector<PVCol> cols;
		for (Pair const& pair : _pairs) {
			for (PVCol col : {pair.x, pair.y}) {
				if (std::find(cols.begin(), cols.end(), col) == cols.end()) {
					cols.push_back(col);
				}
			}
		}

		// One sample size for the whole pass, so the per-column sums and the
		// cross terms describe the same rows -- and so that a wide file does not
		// turn ranking into minutes of work.
		const PVRow sample =
		    PVScatterThumbnail::correlation_sample_size(_view.get_row_count(), _pairs.size());

		std::vector<PVScatterThumbnail::ColumnMoments> computed(cols.size());
		tbb::parallel_for(size_t(0), cols.size(), [&](size_t i) {
			if (cancelled->load()) {
				return;
			}
			computed[i] = PVScatterThumbnail::column_moments(_view, cols[i], sample);
		});

		std::unordered_map<PVCol, PVScatterThumbnail::ColumnMoments> moments;
		for (size_t i = 0; i < cols.size(); i++) {
			moments[cols[i]] = computed[i];
		}

		// at(), not operator[]: the latter may insert, which several threads
		// doing at once would corrupt the map.
		tbb::parallel_for(size_t(0), _pairs.size(), [&](size_t i) {
			if (cancelled->load()) {
				return;
			}
			(*scores)[i] =
			    PVScatterThumbnail::correlation(_view, _pairs[i].x, _pairs[i].y,
			                                    moments.at(_pairs[i].x), moments.at(_pairs[i].y),
			                                    sample);
		});

		QMetaObject::invokeMethod(
		    this,
		    [this, scores, cancelled]() {
			    _correlations_pending = false;
			    // Ranking on a set computed against axes that have since changed
			    // would put the thumbnails in an order that means nothing.
			    if (cancelled->load() or scores->size() != _pairs.size()) {
				    return;
			    }
			    _correlations = *scores;
			    Q_EMIT correlations_ready();
			},
		    Qt::QueuedConnection);
	});
}

void PVParallelView::PVScatterThumbnailsModel::on_selection_changed()
{
	invalidate_thumbnails();
}

void PVParallelView::PVScatterThumbnailsModel::on_output_layer_changed()
{
	invalidate_thumbnails();
}

void PVParallelView::PVScatterThumbnailsModel::on_scaling_changed()
{
	// Scaling rewrites the very values the ranking was computed from, so the
	// order on screen would otherwise stay the one of the previous values --
	// wrong, and with nothing to show that it is.
	const bool ranked = not _correlations.empty();
	_correlations.clear();
	cancel_correlations();

	invalidate_thumbnails();

	if (ranked) {
		compute_correlations();
	}
}

void PVParallelView::PVScatterThumbnailsModel::on_unselected_zombie_visibility_toggled()
{
	// Only the composition changes, but the two layers are not kept once
	// composed, so the thumbnails are rendered again.
	invalidate_thumbnails();
}

void PVParallelView::PVScatterThumbnailsModel::on_axes_combination_about_to_change()
{
	// The pair list is about to name columns that may no longer be in the
	// combination: stop everything reading it before it is rebuilt.
	drain();
	_axes_combination_changing = true;
}

void PVParallelView::PVScatterThumbnailsModel::on_axes_combination_changed()
{
	_axes_combination_changing = false;

	_cache.clear();
	_lru.clear();
	_cached_bytes = 0;
	reset_pairs();
}
