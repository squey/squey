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

#include <pvguiqt/PVAbstractTableModel.h>
#include <pvguiqt/PVAbstractTableView.h>

#include <pvkernel/core/PVUtils.h>
#include <pvkernel/core/squey_assert.h>

#include <QApplication>
#include <QHeaderView>
#include <QKeyEvent>
#include <QScrollBar>

/**
 * Keyboard navigation of PVAbstractTableView over a paginated listing.
 *
 * The listing must hold more than MIN_PAGE_SIZE * SCROLL_SIZE rows so that
 * PVAbstractTableModel really paginates, which is the case the arrow keys have to
 * cope with: view row indexes then only address the shown page and no longer the
 * whole listing.
 */
static constexpr size_t ROW_COUNT = 600000;
static constexpr int ROW_HEIGHT = 20;
static constexpr int SHOWN_ROWS = 10;

namespace
{

/**
 * Minimal concrete model, the navigation only needs the pagination of the abstract one.
 */
class PVTestTableModel : public PVGuiQt::PVAbstractTableModel
{
  public:
	explicit PVTestTableModel(int row_count) : PVAbstractTableModel(row_count) {}

	int columnCount(const QModelIndex& = QModelIndex()) const override { return 1; }

	QVariant data(const QModelIndex& index, int role) const override
	{
		if (role != Qt::DisplayRole) {
			return {};
		}
		return QString::number(rowIndex(index));
	}

	QString export_line(int row, const QString&) const override { return QString::number(row); }
};

/**
 * Counts the selection_commited signals emitted by a PVAbstractTableView.
 */
class PVSelectionCommitedCounter : public QObject
{
  public:
	void on_selection_commited() { _count++; }

	size_t count() const { return _count; }

  private:
	size_t _count = 0;
};

} // namespace

static void press(QWidget& view, int key, Qt::KeyboardModifiers modifiers = Qt::NoModifier)
{
	QKeyEvent event(QEvent::KeyPress, key, modifiers);
	QApplication::sendEvent(&view, &event);
}

/**
 * Position, in the listing, of the row shown on the first line of the view.
 */
static ssize_t first_shown_pos(PVGuiQt::PVAbstractTableView& view, PVTestTableModel& model)
{
	const int first_shown_row = view.rowAt(0);
	PV_ASSERT_VALID(first_shown_row >= 0);
	return model.row_pos(PVRow(first_shown_row));
}

int main(int argc, char** argv)
{
	// Let the test run headless while still allowing to watch it on a real display
	PVCore::setenv("QT_QPA_PLATFORM", "offscreen", 0);

	QApplication app(argc, argv);

	PVTestTableModel model(ROW_COUNT);
	PVGuiQt::PVAbstractTableView view;
	view.horizontalHeader()->hide();
	view.verticalHeader()->setSectionResizeMode(QHeaderView::Fixed);
	view.verticalHeader()->setDefaultSectionSize(ROW_HEIGHT);
	view.setModel(&model);
	// Qualified as PVTableView declares a resize() signal hiding QWidget::resize
	view.QWidget::resize(200, SHOWN_ROWS * ROW_HEIGHT);
	view.show();
	// Let the view compute its geometries, hence its page step and its shown rows
	QApplication::processEvents();

	// Moving the current row must commit and notify as a mouse click does
	PVSelectionCommitedCounter commited;
	PV_ASSERT_VALID(
	    bool(QObject::connect(&view, &PVGuiQt::PVAbstractTableView::selection_commited, &commited,
	                          &PVSelectionCommitedCounter::on_selection_commited)));

	const ssize_t shown_rows = view.verticalScrollBar()->pageStep();
	PV_ASSERT_VALID(shown_rows > 1, "shown_rows", shown_rows);
	PV_VALID(first_shown_pos(view, model), ssize_t(0));

	// No current row until the keyboard navigation starts
	PV_VALID(model.current_row_pos(), ssize_t(-1));

	// The navigation starts on the row shown on the first line
	press(view, Qt::Key_Down);
	PV_VALID(model.current_row_pos(), ssize_t(0));
	PV_VALID(first_shown_pos(view, model), ssize_t(0));
	// The current row is commited in the current selection and notified as a click is
	PV_VALID(commited.count(), size_t(1));
	PV_ASSERT_VALID(model.current_selection().get_line(model.row_pos_to_index(0)));

	// Moving down within the shown rows does not scroll the listing
	for (ssize_t row = 1; row < shown_rows; row++) {
		press(view, Qt::Key_Down);
		PV_VALID(model.current_row_pos(), row);
		PV_VALID(first_shown_pos(view, model), ssize_t(0));
	}

	// Moving down out of the shown rows scrolls one row at a time
	for (ssize_t i = 1; i <= 3; i++) {
		press(view, Qt::Key_Down);
		PV_VALID(model.current_row_pos(), shown_rows - 1 + i);
		PV_VALID(first_shown_pos(view, model), i);
	}

	// Moving back up within the shown rows does not scroll the listing either
	for (ssize_t row = shown_rows + 1; row >= 3; row--) {
		press(view, Qt::Key_Up);
		PV_VALID(model.current_row_pos(), row);
		PV_VALID(first_shown_pos(view, model), ssize_t(3));
	}

	// Then it scrolls back one row at a time, up to the top of the listing
	for (ssize_t i = 2; i >= 0; i--) {
		press(view, Qt::Key_Up);
		PV_VALID(model.current_row_pos(), i);
		PV_VALID(first_shown_pos(view, model), i);
	}
	press(view, Qt::Key_Up);
	PV_VALID(model.current_row_pos(), ssize_t(0));
	PV_VALID(first_shown_pos(view, model), ssize_t(0));

	// Only the current row is selected, the previously selected ones are released
	PV_ASSERT_VALID(model.is_selected(model.index(int(model.current_page()), 0)));
	PV_ASSERT_VALID(not model.is_selected(model.index(int(model.current_page()) + 1, 0)));
	PV_VALID(model.current_selection().bit_count(), size_t(1));

	// Shift extends the selection from the current row instead of restarting it
	press(view, Qt::Key_Down, Qt::ShiftModifier);
	press(view, Qt::Key_Down, Qt::ShiftModifier);
	PV_VALID(model.current_row_pos(), ssize_t(2));
	for (int row = 0; row <= 2; row++) {
		PV_ASSERT_VALID(model.is_selected(model.index(int(model.current_page()) + row, 0)), "row",
		                row);
	}
	PV_ASSERT_VALID(not model.is_selected(model.index(int(model.current_page()) + 3, 0)));
	PV_VALID(model.current_selection().bit_count(), size_t(3));

	// Control scrolls the listing without moving the current row nor touching the selection
	const size_t commited_before_scroll = commited.count();
	press(view, Qt::Key_Down, Qt::ControlModifier);
	PV_VALID(model.current_row_pos(), ssize_t(2));
	PV_VALID(first_shown_pos(view, model), ssize_t(1));
	PV_VALID(commited.count(), commited_before_scroll);

	// The current row is brought back into the shown rows on the next move
	press(view, Qt::Key_Up);
	PV_VALID(model.current_row_pos(), ssize_t(1));
	PV_VALID(first_shown_pos(view, model), ssize_t(1));

	// The current row does not go past the end of the listing
	model.reset_selection();
	view.verticalScrollBar()->setValue(view.verticalScrollBar()->maximum());
	press(view, Qt::Key_Down);
	const ssize_t last_page_first_pos = model.current_row_pos();
	PV_ASSERT_VALID(last_page_first_pos >= ssize_t(ROW_COUNT) - shown_rows - 1, "current_row_pos",
	                last_page_first_pos, "shown_rows", shown_rows);
	for (ssize_t i = 0; i < 2 * shown_rows; i++) {
		press(view, Qt::Key_Down);
	}
	PV_VALID(model.current_row_pos(), ssize_t(ROW_COUNT) - 1);
	// And the listing is scrolled down to it. The very last row may only be partially
	// shown as the listing can't be scrolled past its end and the viewport height is
	// not a multiple of the row height, hence the inclusive upper bound.
	const ssize_t first_pos = first_shown_pos(view, model);
	PV_ASSERT_VALID(first_pos <= model.current_row_pos() and
	                    model.current_row_pos() <= first_pos + shown_rows,
	                "first_pos", first_pos, "current_row_pos", model.current_row_pos());

	// Sorting or filtering the listing resets the selection, which invalidates the current
	// row as row positions do not refer to the same rows anymore
	model.reset_selection();
	PV_VALID(model.current_row_pos(), ssize_t(-1));

	return 0;
}
