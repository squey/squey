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

// The dialog of an input type shows in the import page as what it asks for an import: its buttons
// hide, the button that accepted it tells whether it is ready, and Import clicks that button,
// which runs the checks of the dialog. Escape leaves it in the page, Enter goes on to the page,
// which imports then, and it is rejected as it goes, as closing its window did.

#include <pvkernel/core/squey_assert.h>
#include <pvkernel/widgets/PVDialogImportWidget.h>

#include <QApplication>
#include <QDialog>
#include <QDialogButtonBox>
#include <QKeyEvent>
#include <QLineEdit>
#include <QPointer>
#include <QPushButton>
#include <QTest>
#include <QVBoxLayout>

namespace
{

/** Gets the keys the widgets in it leave, as the import page does. */
class Page : public QWidget
{
  public:
	int enters = 0;

  protected:
	void keyPressEvent(QKeyEvent* event) override
	{
		if (event->key() == Qt::Key_Return) {
			enters++;
			return;
		}
		QWidget::keyPressEvent(event);
	}
};

} // namespace

int main(int argc, char** argv)
{
	QApplication app(argc, argv); // argv carries "-platform offscreen"
	Page page;
	page.show();

	// Accepted only once it names something, as the checks of a dialog do.
	auto* dialog = new QDialog;
	auto* name = new QLineEdit;
	auto* buttons = new QDialogButtonBox(QDialogButtonBox::Open | QDialogButtonBox::Cancel);
	auto* layout = new QVBoxLayout(dialog);
	layout->addWidget(name);
	layout->addWidget(buttons);
	QObject::connect(buttons, &QDialogButtonBox::accepted, dialog, [dialog, name]() {
		if (not name->text().isEmpty()) {
			dialog->accept();
		}
	});
	int rejected = 0;
	QObject::connect(dialog, &QDialog::rejected, [&]() { rejected++; });

	QStringList paths;
	int imports = 0;
	QAbstractButton* open = buttons->button(QDialogButtonBox::Open);
	auto content = std::make_unique<PVWidgets::PVDialogImportWidget>(
	    dialog,
	    PVWidgets::PVDialogImportWidget::Parts{
	        .accept = open,
	        .buttons = {buttons},
	        .sources =
	            [&]() {
		            imports++;
		            return std::vector<PVWidgets::PVImportWidget::source_t>(1);
	            },
	        .add_paths = [&](QStringList const& given) { paths = given; }},
	    &page);
	content->show();
	PV_ASSERT_VALID(dialog->isVisible() and not dialog->isWindow());
	PV_ASSERT_VALID(not buttons->isVisible());
	PV_ASSERT_VALID(not content->fills_page());

	int ready_changes = 0;
	QObject::connect(content.get(), &PVWidgets::PVImportWidget::ready_changed,
	                 [&]() { ready_changes++; });
	PV_ASSERT_VALID(content->is_ready());
	open->setEnabled(false);
	PV_ASSERT_VALID(not content->is_ready());
	PV_VALID(ready_changes, 1);
	open->setEnabled(true);

	PV_VALID(content->sources().size(), size_t(0));
	PV_VALID(imports, 0);
	name->setText("something");
	PV_VALID(content->sources().size(), size_t(1));
	PV_VALID(imports, 1);
	PV_ASSERT_VALID(dialog->isVisible());

	QTest::keyClick(name, Qt::Key_Escape);
	PV_ASSERT_VALID(dialog->isVisible());
	PV_VALID(rejected, 0);
	QTest::keyClick(name, Qt::Key_Return);
	PV_VALID(page.enters, 1);
	PV_VALID(imports, 1);

	content->set_paths({"capture.pcap"});
	PV_ASSERT_VALID(paths == QStringList{"capture.pcap"});
	// As files dropped on the page.
	PV_ASSERT_VALID(content->takes_files());
	content->add_paths({"other.pcap"});
	PV_ASSERT_VALID(paths == QStringList{"other.pcap"});

	const QPointer<QDialog> owned = dialog;
	content.reset();
	PV_VALID(rejected, 1);
	PV_ASSERT_VALID(owned.isNull());

	return 0;
}
