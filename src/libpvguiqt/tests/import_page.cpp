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
// buttons for every type: Import once the content is ready, Cancel, and the data collection to
// import into, in a column at the center of a wide window. It runs the long tasks of the content
// out of the GUI thread, with their progress and a button to cancel them, and tells what a task
// that failed said.

#include <pvguiqt/PVImportPage.h>

#include <pvkernel/core/PVTaskProgress.h>
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/widgets/PVImportWidget.h>

#include <QApplication>
#include <QComboBox>
#include <QLabel>
#include <QPointer>
#include <QPushButton>
#include <QTest>

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

QPushButton* button(QWidget& parent, QString const& name)
{
	return parent.findChild<QPushButton*>(name);
}

QPushButton* progress_cancel(PVGuiQt::PVImportPage& page)
{
	for (QPushButton* button :
	     page.findChild<QWidget*>("import_progress")->findChildren<QPushButton*>()) {
		if (button->text() == "Cancel" and button->isVisible()) {
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
	int cancels = 0;
	int finished_tasks = 0;
	QObject::connect(&page, &PVGuiQt::PVImportPage::import_requested, [&]() { imports++; });
	QObject::connect(&page, &PVGuiQt::PVImportPage::cancel_requested, [&]() { cancels++; });
	QObject::connect(&page, &PVGuiQt::PVImportPage::task_finished, [&]() { finished_tasks++; });

	QPointer<Content> content = new Content;
	page.set_content(content, "Import: Content", {"First collection", "Second collection"}, 1);
	auto* collection = page.findChild<QComboBox*>("import_collection");
	PV_VALID(collection->count(), 3);
	PV_VALID(page.chosen_collection(), 1);
	collection->setCurrentIndex(2);
	PV_VALID(page.chosen_collection(), -1);

	QPushButton* import = button(page, "import_import");
	QPushButton* cancel = button(page, "import_cancel");

	// What the page shows keeps to a column at the center of a wide page, with the buttons right
	// under it, and takes the whole width of a narrow one.
	QWidget* column = content->parentWidget();
	QWidget* area = column->parentWidget();
	page.resize(3000, 1500);
	PV_ASSERT_VALID(QTest::qWaitFor([&]() { return area->width() > 2500; }, 10000));
	PV_ASSERT_VALID(column->width() < area->width() / 2);
	PV_ASSERT_VALID(std::abs(2 * column->x() + column->width() - area->width()) <= 1);
	PV_ASSERT_VALID(import->mapTo(&page, QPoint(0, import->height())).y() < page.height() / 4);
	page.resize(800, 1500);
	PV_ASSERT_VALID(QTest::qWaitFor([&]() { return area->width() < 900; }, 10000));
	PV_VALID(column->width(), area->width());

	PV_ASSERT_VALID(import->isVisible() and not import->isEnabled());
	PV_ASSERT_VALID(cancel->isVisible() and cancel->isEnabled());
	PV_ASSERT_VALID(not button(page, "import_close")->isVisible());
	content->set_ready(true);
	PV_ASSERT_VALID(import->isEnabled());
	import->click();
	cancel->click();
	PV_VALID(imports, 1);
	PV_VALID(cancels, 1);

	// A task shows its progress, and keeps the content and its import away until it ends.
	std::optional<bool> completed;
	content->run_task(
	    "Reading",
	    [](PVCore::PVTaskProgress& progress) {
		    progress.set_maximum(10);
		    progress.set_value(4);
		    progress.set_status("Four of ten");
		    while (not progress.canceled()) {
			    std::this_thread::sleep_for(std::chrono::milliseconds(10));
		    }
	    },
	    [&completed](bool done) { completed = done; });
	PV_ASSERT_VALID(page.task_running());
	PV_ASSERT_VALID(not content->isEnabled() and not import->isEnabled() and
	                not cancel->isEnabled());
	auto* details = page.findChild<QLabel*>("import_details");
	PV_ASSERT_VALID(QTest::qWaitFor([&]() { return details->text() == "Four of ten"; }, 10000));
	PV_ASSERT_VALID(progress_cancel(page) != nullptr);
	progress_cancel(page)->click();
	PV_ASSERT_VALID(QTest::qWaitFor([&]() { return completed.has_value(); }, 10000));
	PV_VALID(*completed, false);
	PV_ASSERT_VALID(not page.task_running());
	PV_ASSERT_VALID(content->isEnabled() and import->isEnabled() and cancel->isEnabled());
	PV_VALID(finished_tasks, 1);

	PV_VALID(*run(*content, [](PVCore::PVTaskProgress&) {}), true);
	PV_VALID(messages_of(page), std::string());
	// What a task throws is told, as text.
	PV_VALID(*run(*content, [](PVCore::PVTaskProgress&) { throw std::runtime_error("No <file>"); }),
	         false);
	PV_VALID(messages_of(page), std::string("No &lt;file&gt;"));
	PV_VALID(finished_tasks, 3);

	page.clear_content();
	PV_ASSERT_VALID(not import->isVisible() and not cancel->isVisible());
	PV_ASSERT_VALID(QTest::qWaitFor([&]() { return content.isNull(); }, 10000));

	return 0;
}
