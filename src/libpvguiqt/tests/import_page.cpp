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

// The import page shows what an input type asks for an import, its content, with the same
// buttons for every type: Import once the content is ready, Cancel, Back, and the data
// collection to import into, in a column at the center of a wide window. It runs the long tasks
// of the content out of the GUI thread, with their progress and a button to cancel them, and
// tells what a task that failed said.

#include <pvguiqt/PVImportPage.h>

#include <pvkernel/core/PVTaskProgress.h>
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/widgets/PVImportWidget.h>

#include <QApplication>
#include <QComboBox>
#include <QDropEvent>
#include <QLabel>
#include <QMimeData>
#include <QPointer>
#include <QPushButton>
#include <QTest>
#include <QUrl>

#include <chrono>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

namespace
{

class Content : public PVWidgets::PVImportWidget
{
  public:
	/** Takes the height it gets, as a content holding lists does. */
	Content() { setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding); }

	bool is_ready() const override { return _ready; }
	std::vector<source_t> sources() const override { return {}; }

	void set_ready(bool ready)
	{
		_ready = ready;
		Q_EMIT ready_changed();
	}

	using PVWidgets::PVImportWidget::run_task;

  private:
	bool _ready = false;
};

class FillingContent : public Content
{
  public:
	bool fills_page() const override { return true; }
};

class FilesContent : public Content
{
  public:
	bool takes_files() const override { return true; }
	void add_paths(QStringList const& paths) override { added += paths; }

	QStringList added;
};

/** Drags @p mime over @p widget, and drops it there if it accepts it: whether it took it. */
bool drop(QWidget& widget, QMimeData const& mime)
{
	const QPointF center = widget.rect().center();
	QDragEnterEvent enter(center, Qt::CopyAction, &mime, Qt::LeftButton, Qt::NoModifier);
	QCoreApplication::sendEvent(&widget, &enter);
	if (not enter.isAccepted()) {
		return false;
	}
	PV_ASSERT_VALID(widget.findChild<QWidget*>("import_drop_overlay")->isVisible());
	QDropEvent dropped(center, Qt::CopyAction, &mime, Qt::LeftButton, Qt::NoModifier);
	QCoreApplication::sendEvent(&widget, &dropped);
	PV_ASSERT_VALID(not widget.findChild<QWidget*>("import_drop_overlay")->isVisible());
	return dropped.isAccepted();
}

QPushButton* button(QWidget& parent, QString const& name)
{
	return parent.findChild<QPushButton*>(name);
}

QPushButton* progress_cancel(PVGuiQt::PVImportPage& page)
{
	for (QPushButton* button :
	     page.findChild<QWidget*>("import_progress")->findChildren<QPushButton*>()) {
		if (button->text().trimmed() == "Cancel" and button->isVisible()) {
			return button;
		}
	}
	return nullptr;
}

std::string messages_of(PVGuiQt::PVImportPage& page)
{
	std::string texts;
	for (const QLabel* label : page.findChildren<QLabel*>()) {
		if (label->parentWidget()->objectName() == "import_message") {
			texts += label->text().toStdString();
		}
	}
	return texts;
}

void until_canceled(PVCore::PVTaskProgress& progress)
{
	while (not progress.canceled()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
}

/** Runs @p work as a task of @p content, and waits for its end: whether it completed. */
std::optional<bool> run(Content& content, std::function<void(PVCore::PVTaskProgress&)> work)
{
	std::optional<bool> completed;
	content.run_task("Working", std::move(work), [&](bool done) { completed = done; });
	PV_ASSERT_VALID(QTest::qWaitFor([&]() { return completed.has_value(); }, 10000));
	return completed;
}

} // namespace

int main(int argc, char** argv)
{
	QApplication app(argc, argv); // argv carries "-platform offscreen"

	PVGuiQt::PVImportPage page;
	page.show();
	int imports = 0;
	int closes = 0;
	int backs = 0;
	int finished_tasks = 0;
	QObject::connect(&page, &PVGuiQt::PVImportPage::import_requested, [&]() { imports++; });
	QObject::connect(&page, &PVGuiQt::PVImportPage::close_requested, [&]() { closes++; });
	QObject::connect(&page, &PVGuiQt::PVImportPage::back_requested, [&]() { backs++; });
	QObject::connect(&page, &PVGuiQt::PVImportPage::task_finished, [&]() { finished_tasks++; });

	QPointer<Content> content = new Content;
	page.set_content(content, "Import: Content", {}, {"First collection", "Second collection"}, 1);
	auto* collection = page.findChild<QComboBox*>("import_collection");
	PV_ASSERT_VALID(collection->isVisible());
	PV_VALID(collection->count(), 3);
	PV_VALID(page.chosen_collection(), 1);
	collection->setCurrentIndex(2);
	PV_VALID(page.chosen_collection(), -1);

	QPushButton* import = button(page, "import_import");
	QPushButton* cancel = button(page, "import_cancel");

	// What the page shows keeps to a sheet at the center of a wide page, with the buttons right
	// under it, and takes the width of a narrow one. The sheet holds a number of characters: the
	// wide page is three times as wide as the sheet may be, whatever the font, which the
	// offscreen platform of the Windows CI has none of, drawing boxes wider than letters.
	QWidget* column = page.findChild<QWidget*>("import_sheet");
	PV_ASSERT_VALID(column->isAncestorOf(content) and column->isAncestorOf(import));
	PV_ASSERT_VALID(column->maximumWidth() < QWIDGETSIZE_MAX);
	const int wide = 3 * column->maximumWidth();
	page.resize(wide, 1500);
	PV_ASSERT_VALID(QTest::qWaitFor([&]() { return page.width() > wide * 5 / 6; }, 10000));
	PV_ASSERT_VALID(column->width() < page.width() / 2);
	PV_ASSERT_VALID(std::abs(2 * column->x() + column->width() - page.width()) <= 2);
	PV_ASSERT_VALID(import->mapTo(&page, QPoint(0, import->height())).y() < page.height() / 4);
	page.resize(800, 1500);
	PV_ASSERT_VALID(QTest::qWaitFor([&]() { return page.width() < 900; }, 10000));
	PV_ASSERT_VALID(column->width() > page.width() * 9 / 10);

	PV_ASSERT_VALID(import->isVisible() and not import->isEnabled());
	PV_ASSERT_VALID(cancel->isVisible() and cancel->isEnabled());
	PV_ASSERT_VALID(not button(page, "import_close")->isVisible());
	content->set_ready(true);
	PV_ASSERT_VALID(import->isEnabled());
	import->click();
	cancel->click();
	PV_VALID(imports, 1);
	PV_VALID(closes, 1);
	// Enter imports too, as it accepts a dialog, but on a button, which it does not click.
	QTest::keyClick(&page, Qt::Key_Return);
	PV_ASSERT_VALID(QTest::qWaitFor([&]() { return imports == 2; }, 10000));
	cancel->setFocus();
	QTest::keyClick(cancel, Qt::Key_Return);
	QTest::qWait(300);
	PV_VALID(imports, 2);
	PV_VALID(closes, 1);
	QPushButton* back = button(page, "import_back");
	back->click();
	PV_VALID(backs, 1);
	page.set_back_enabled(false);
	PV_ASSERT_VALID(not back->isEnabled());
	page.set_back_enabled(true);

	// A task shows its progress, and keeps the content and its import away until it ends, but not
	// the Cancel of the page, which stops it first.
	std::optional<bool> completed;
	content->run_task(
	    "Reading",
	    [](PVCore::PVTaskProgress& progress) {
		    progress.set_maximum(10);
		    progress.set_value(4);
		    progress.set_status("Four of ten");
		    until_canceled(progress);
	    },
	    [&completed](bool done) { completed = done; });
	PV_ASSERT_VALID(page.task_running());
	PV_ASSERT_VALID(not content->isEnabled() and not import->isEnabled() and cancel->isEnabled());
	auto* details = page.findChild<QLabel*>("import_details");
	PV_ASSERT_VALID(QTest::qWaitFor([&]() { return details->text() == "Four of ten"; }, 10000));
	PV_ASSERT_VALID(progress_cancel(page) != nullptr);
	progress_cancel(page)->click();
	PV_ASSERT_VALID(QTest::qWaitFor([&]() { return completed.has_value(); }, 10000));
	PV_VALID(*completed, false);
	PV_ASSERT_VALID(not page.task_running());
	PV_ASSERT_VALID(content->isEnabled() and import->isEnabled() and cancel->isEnabled());
	PV_VALID(finished_tasks, 1);

	// As closing the page asks.
	completed.reset();
	content->run_task("Reading", until_canceled, [&completed](bool done) { completed = done; });
	page.cancel_task();
	PV_ASSERT_VALID(QTest::qWaitFor([&]() { return completed.has_value(); }, 10000));
	PV_VALID(*completed, false);
	PV_VALID(finished_tasks, 2);

	PV_VALID(*run(*content, [](PVCore::PVTaskProgress&) {}), true);
	PV_VALID(messages_of(page), std::string());
	// What a task throws is told, as text.
	PV_VALID(*run(*content, [](PVCore::PVTaskProgress&) { throw std::runtime_error("No <file>"); }),
	         false);
	PV_VALID(messages_of(page), std::string("No &lt;file&gt;"));
	PV_VALID(finished_tasks, 4);

	page.clear_content();
	PV_ASSERT_VALID(not import->isVisible() and not cancel->isVisible());
	PV_ASSERT_VALID(QTest::qWaitFor([&]() { return content.isNull(); }, 10000));

	// But a content that fills the page takes its width and its height.
	const QPointer<FillingContent> filling = new FillingContent;
	page.set_content(filling, "Import: Filling", {}, {}, -1);
	// With no data collection to choose from, the import makes a new one without asking.
	PV_ASSERT_VALID(not collection->isVisible());
	PV_VALID(page.chosen_collection(), -1);
	page.resize(wide, 1500);
	PV_ASSERT_VALID(
	    QTest::qWaitFor([&]() { return column->width() > page.width() * 9 / 10; }, 10000));
	PV_ASSERT_VALID(filling->height() > page.height() / 2);
	// But not the progress of its import, which keeps to the column of the others.
	page.set_content_shown(false);
	page.start_progress("Importing");
	PV_ASSERT_VALID(QTest::qWaitFor(
	    [&]() {
		    return column->width() < page.width() / 2 and column->height() < page.height() / 2;
	    },
	    10000));
	page.end_progress();
	page.set_content_shown(true);
	PV_ASSERT_VALID(
	    QTest::qWaitFor([&]() { return column->width() > page.width() * 9 / 10; }, 10000));
	page.clear_content();
	PV_ASSERT_VALID(QTest::qWaitFor([&]() { return column->width() < page.width() / 2; }, 10000));

	// Local files dropped on the page go to a content that takes files, while no task runs.
	QMimeData files;
	files.setUrls({QUrl::fromLocalFile("/data/one.csv"), QUrl::fromLocalFile("/data/two.csv")});
	PV_ASSERT_VALID(not drop(page, files));
	const QPointer<FilesContent> taking = new FilesContent;
	page.set_content(taking, "Import: Files", {}, {}, -1);
	PV_ASSERT_VALID(drop(page, files));
	PV_ASSERT_VALID(taking->added == QStringList({"/data/one.csv", "/data/two.csv"}));
	QMimeData remote;
	remote.setUrls({QUrl::fromLocalFile("/data/three.csv"), QUrl("https://squey.org/four.csv")});
	PV_ASSERT_VALID(not drop(page, remote));
	completed.reset();
	taking->run_task("Reading", until_canceled, [&completed](bool done) { completed = done; });
	PV_ASSERT_VALID(not drop(page, files));
	page.cancel_task();
	PV_ASSERT_VALID(QTest::qWaitFor([&]() { return completed.has_value(); }, 10000));
	PV_VALID(taking->added.size(), qsizetype(2));
	page.clear_content();

	return 0;
}
