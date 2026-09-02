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

#ifndef PVPARALLELVIEW_PVSCATTERTHUMBNAILSMODEL_H
#define PVPARALLELVIEW_PVSCATTERTHUMBNAILSMODEL_H

#include <pvbase/types.h>

#include <pvkernel/core/PVDisconnector.h>

#include <pvparallelview/PVScatterThumbnail.h>

#include <tbb/task_group.h>

#include <QAbstractListModel>
#include <QPixmap>
#include <QString>

#include <atomic>
#include <deque>
#include <list>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Squey
{
class PVView;
} // namespace Squey

namespace PVParallelView
{

/**
 * One item per distinct pair of axes of the view's current axes combination,
 * each rendered as a scatter thumbnail.
 *
 * Thumbnails are rendered lazily: data() schedules a render for an item it has
 * no image for and hands back a placeholder, then emits dataChanged once the
 * render lands. Since a QListView only ever queries the items it is about to
 * paint, this alone keeps a gallery of hundreds of pairs to the handful of
 * renders the user can actually see, with no viewport tracking of our own.
 *
 * The renders run on a TBB task group, capped so that a fast scroll cannot
 * queue an unbounded amount of work, and served newest-request-first: what the
 * user is looking at now matters more than what they scrolled past.
 *
 * This view is autonomous -- it needs no zone tree, so no PVViewRenderingContext
 * -- and therefore subscribes straight to the Squey::PVView model, as
 * PVSeriesViewWidget does and as the connection rule in PVViewRenderingContext.h
 * prescribes.
 */
class PVScatterThumbnailsModel : public QAbstractListModel
{
	Q_OBJECT

  public:
	enum Roles {
		//! double in [0, 1], ranks pairs by how structured they look.
		CorrelationRole = Qt::UserRole + 1,
		//! bool, true when the thumbnail was rendered from a sample of the rows.
		SampledRole,
	};

	//! Range of the toolbar's size slider; any value in it renders.
	constexpr static int min_size = 48;
	constexpr static int max_size = 320;
	constexpr static int default_size = 128;
	//! Resolution of the enlarged preview shown on hover.
	constexpr static int preview_size = 512;

	explicit PVScatterThumbnailsModel(Squey::PVView& view, QObject* parent = nullptr);
	// noexcept is explicit because tbb::task_group's destructor is not, which
	// would otherwise make this one laxer than QAbstractListModel's.
	~PVScatterThumbnailsModel() noexcept override;

  public:
	int rowCount(QModelIndex const& parent = QModelIndex()) const override;
	QVariant data(QModelIndex const& index, int role = Qt::DisplayRole) const override;

  public:
	Squey::PVView& lib_view() const { return _view; }

	int thumbnail_size() const { return _size; }
	void set_thumbnail_size(int size);

	std::pair<PVCol, PVCol> axes_at(int row) const;
	QString label_at(int row) const;

	/**
	 * Swap the two axes of one pair, so the user can look at it the other way
	 * round without leaving the gallery.
	 */
	void swap_axes(int row);

	/**
	 * Render one pair at an arbitrary size for the hover preview, in the
	 * background; preview_ready() carries the result.
	 *
	 * A preview is several times the area of a thumbnail, so on a large file it
	 * takes long enough that rendering it on the UI thread freezes the whole
	 * application under the pointer. Only the latest request is honoured: the
	 * previous one is cancelled, since the pointer has moved on.
	 */
	void request_preview(int row, int size);

	/**
	 * Render one pair right now. Used where the caller has nowhere to wait, as
	 * when copying a thumbnail to the clipboard.
	 */
	QPixmap render_preview(int row, int size) const;

	/**
	 * Score every pair so the gallery can be ranked by correlation.
	 *
	 * Runs in the background; correlations_ready() is emitted once the whole
	 * set is known, since a partial ranking would keep re-ordering under the
	 * user's pointer.
	 */
	void compute_correlations();
	bool has_correlations() const { return not _correlations.empty(); }

	/**
	 * Cancel every in-flight render and wait for it to actually stop.
	 *
	 * Must be called before anything a running render reads goes away -- which
	 * is why the model does it in its destructor and on model teardown.
	 */
	void drain();

	/**
	 * Cancel every in-flight render without waiting for it.
	 *
	 * What invalidations use: their results are discarded through the
	 * cancellation flag anyway, and blocking the UI thread on every selection
	 * change -- which is once per frame while a selection rectangle is being
	 * dragged -- would make the whole application stutter.
	 */
	void cancel_renders();

	//! Cancel a running correlation pass without touching the renders.
	void cancel_correlations();

  Q_SIGNALS:
	void correlations_ready();

	//! The background render asked for by request_preview() has landed.
	void preview_ready(int row, QPixmap const& pixmap);

	/**
	 * Every queued render is done and none is running.
	 *
	 * The gallery repaints on this: a repaint only asks for the items it is
	 * about to paint, and the request queue is bounded, so a screenful larger
	 * than that bound leaves some visible items unrequested. Repainting once
	 * the queue drains asks for whatever is still missing, and does nothing at
	 * all once everything visible is cached.
	 */
	void renders_idle();

  private:
	struct Pair {
		PVCol x;
		PVCol y;
	};

	//! Rebuild the pair list from the current axes combination.
	void reset_pairs();
	//! Drop every rendered thumbnail (colours, scaling or size changed).
	void invalidate_thumbnails();

	void request_render(int row) const;
	void schedule_renders() const;
	void store_render(int row, PVScatterThumbnailImages const& images);

	//! Compose the two layers the way PVScatterView::drawBackground does.
	QPixmap compose(PVScatterThumbnailImages const& images) const;

	void rebuild_placeholder();
	void touch_cache(int row) const;
	void trim_cache() const;

	// Model signal handlers
	void on_selection_changed();
	void on_output_layer_changed();
	void on_scaling_changed();
	void on_unselected_zombie_visibility_toggled();
	void on_axes_combination_about_to_change();
	void on_axes_combination_changed();

  private:
	Squey::PVView& _view;

	std::vector<Pair> _pairs;
	std::vector<double> _correlations;
	//! A scoring pass is already running; a second would redo its work.
	bool _correlations_pending = false;

	int _size = default_size;

	struct CacheEntry {
		QPixmap pixmap;
		bool sampled = false;
		std::list<int>::iterator lru;
		size_t bytes = 0;
	};
	//! mutable: data() is const but must be able to serve and fill the cache.
	mutable std::unordered_map<int, CacheEntry> _cache;
	mutable std::list<int> _lru;

	/**
	 * Stand-in returned for a thumbnail not rendered yet.
	 *
	 * Kept rather than built on demand: data() is asked for it once per
	 * not-yet-rendered visible item on every repaint, which while scrolling is
	 * hundreds of pixmap allocations a second.
	 */
	QPixmap _placeholder;

	mutable std::deque<int> _pending;
	/**
	 * Rows a render is queued or running for.
	 *
	 * A row leaves _pending when its render starts but has no image yet, so
	 * without this a repaint in between would queue the very same render a
	 * second time.
	 */
	mutable std::unordered_set<int> _in_flight;
	mutable size_t _running = 0;
	/**
	 * A render landed since renders_idle() was last emitted.
	 *
	 * Guards the repaint loop: without it, a thumbnail that cannot be rendered
	 * at all would be re-requested by the repaint its own failure triggered.
	 */
	mutable bool _stored_since_idle = false;

	/**
	 * Cancellation flag shared with the renders currently in flight.
	 *
	 * Replaced (and the old one raised) whenever pending work becomes stale, so
	 * that cancelling an obsolete batch never cancels the batch replacing it.
	 */
	mutable std::shared_ptr<std::atomic<bool>> _cancelled;

	/**
	 * Cancellation flag for the correlation pass, kept apart from the render one.
	 *
	 * A correlation does not depend on the selection, so a selection change --
	 * which cancels every render -- must not take the ranking down with it. It
	 * used to share _cancelled, so moving the selection while the ranking ran
	 * killed it for good and left the gallery on "Ranking pairs..." forever.
	 */
	mutable std::shared_ptr<std::atomic<bool>> _correlations_cancelled;

	//! Cancels the in-flight hover preview when the pointer moves to another item.
	std::shared_ptr<std::atomic<bool>> _preview_cancelled;

	mutable tbb::task_group _tasks;

	//! Renders in flight at once, so a fast scroll cannot queue unbounded work.
	size_t _max_concurrent_renders;
	/**
	 * Cache ceiling, in bytes rather than in thumbnails: one large thumbnail
	 * costs 45 times one small one, so a count would either starve the small
	 * sizes or let the large ones run to hundreds of megabytes.
	 */
	size_t _max_cached_bytes;
	mutable size_t _cached_bytes = 0;

	bool _axes_combination_changing = false;

	PVCore::PVDisconnector _selection_changed_connection;
	PVCore::PVDisconnector _output_selection_connection;
	PVCore::PVDisconnector _output_layer_connection;
	PVCore::PVDisconnector _scaling_connection;
	PVCore::PVDisconnector _unselected_zombie_connection;
	PVCore::PVDisconnector _axes_comb_about_to_change_connection;
	PVCore::PVDisconnector _axes_comb_changed_connection;
};

} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVSCATTERTHUMBNAILSMODEL_H
