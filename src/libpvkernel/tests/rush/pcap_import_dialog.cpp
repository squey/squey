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

// The dialog importing captures, used as a user would.
//
// It tells which protocols and fields the captures hold, and finds those tshark knows: a
// field checked becomes a column, which the preview shows as the import extracts it. A wrong
// display filter keeps the import from starting. What is changed of a profile Squey ships is
// kept as a draft, until it is saved as a profile of the user: the dialog opens again as it
// was left. Profiles earlier versions saved show their columns.

#include <pvkernel/core/PVConfig.h>
#include <pvkernel/core/PVUtils.h>
#include <pvkernel/core/squey_assert.h>

#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap/shell.h"
#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap/ws.h"
#include "../../plugins/input-types/pcap/pcap-gui/FieldsPanel.h"
#include "../../plugins/input-types/pcap/pcap-gui/ImportDialog.h"

#include <QApplication>
#include <QComboBox>
#include <QDir>
#include <QFile>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QPushButton>
#include <QTabBar>
#include <QTabWidget>
#include <QTableView>
#include <QTest>
#include <QToolButton>
#include <QTreeView>

#include <iostream>

namespace
{

// Milliseconds. The first frames only show once tshark has listed the fields it knows, which
// takes it half a minute on a loaded runner of the CI.
constexpr int PATIENCE = 120000;

template <typename T>
T* part(QWidget& dialog, const char* name)
{
	T* found = dialog.findChild<T*>(name);
	PV_ASSERT_VALID(found != nullptr, "the part", name);
	return found;
}

void wait_for(const char* what, const std::function<bool()>& done)
{
	PV_ASSERT_VALID(QTest::qWaitFor(done, PATIENCE), "waited in vain for", what);
}

QString text_of(QWidget& dialog, const char* label)
{
	return part<QLabel>(dialog, label)->text();
}

QStringList columns(QWidget& dialog)
{
	QStringList names;
	const QListWidget* list = part<QListWidget>(dialog, "columns_list");
	for (int i = 0; i < list->count(); i++) {
		names << list->item(i)->data(Qt::UserRole).toString();
	}
	return names;
}

void choose_profile(QWidget& dialog, const QString& name)
{
	QListWidget* list = part<QListWidget>(dialog, "profiles_list");
	const QList<QListWidgetItem*> found = list->findItems(name, Qt::MatchExactly);
	PV_ASSERT_VALID(not found.isEmpty(), "the profile", name.toStdString());
	list->setCurrentItem(found.front());
}

//! What the preview shows of a column, row by row.
QStringList preview_column(QWidget& dialog, const QString& column)
{
	const QAbstractItemModel* model = part<QTableView>(dialog, "preview_table")->model();
	QStringList values;
	for (int c = 0; c < model->columnCount(); c++) {
		if (model->headerData(c, Qt::Horizontal, Qt::DisplayRole).toString() == column) {
			for (int r = 0; r < model->rowCount(); r++) {
				values << model->index(r, c).data().toString();
			}
		}
	}
	return values;
}

} // namespace

int main(int argc, char** argv)
{
	// Before what is read below: it takes the options of Qt out of the arguments.
	QApplication application(argc, argv);
	if (argc < 4) {
		std::cerr << "usage: " << argv[0]
		          << " <profiles Squey ships> <pvconfig.ini> <capture of HTTP>" << std::endl;
		return 1;
	}
	PVCore::setenv("SQUEY_PCAP_PROFILES_PATH", argv[1], 1);
	// The configuration of the user, which PVConfig reads, then that of the profiles.
	QDir().mkpath(PVCore::PVConfig::user_dir());
	QFile::remove(PVCore::PVConfig::user_path());
	PV_VALID(QFile::copy(argv[2], PVCore::PVConfig::user_path()), true);
	QDir(QString::fromStdString(pvpcap::get_user_conf_dir())).removeRecursively();
	if (pvpcap::ws_get_version().empty()) {
		std::cerr << "tshark is not available, skipping" << std::endl;
		return 77;
	}
	const QString capture = QFileInfo(argv[3]).absoluteFilePath();

	std::cout << "the fields of the capture" << std::endl;
	{
		PVPcapsicum::ImportDialog dialog({capture});
		dialog.show();
		wait_for("the first frames",
		         [&]() { return text_of(dialog, "fields_status").contains("first 43 frames"); });
		// The protocols the capture holds, with their share of its frames.
		const QAbstractItemModel* fields = part<QTreeView>(dialog, "fields_tree")->model();
		QStringList protocols;
		for (int row = 0; row < fields->rowCount(); row++) {
			protocols << fields->index(row, PVPcapsicum::FieldTreeModel::NAME).data().toString();
		}
		PV_ASSERT_VALID(protocols.contains("HTTP") and protocols.contains("DNS"), "the protocols",
		                protocols.join(" ").toStdString());
		PV_VALID(text_of(dialog, "import_summary").startsWith("1 capture"), true);

		choose_profile(dialog, "HTTP");
		PV_VALID(columns(dialog).size(), qsizetype(27));
		PV_VALID(part<QPushButton>(dialog, "revert_profile_button")->isVisible(), false);

		std::cout << "a field searched, and checked" << std::endl;
		QLineEdit* search = part<QLineEdit>(dialog, "fields_search");
		search->setText("http.host");
		wait_for("the search", [&]() {
			return fields->rowCount() != 0 and
			       fields->index(0, PVPcapsicum::FieldTreeModel::NAME)
			               .data(PVPcapsicum::FieldTreeModel::FIELD_ROLE)
			               .toString() == "http.host";
		});
		// Enter checks it, rather than importing.
		QTest::keyClick(search, Qt::Key_Return);
		PV_VALID(dialog.isVisible(), true);
		PV_VALID(columns(dialog).last().toStdString(), std::string("http.host"));
		PV_VALID(
		    fields->index(0, PVPcapsicum::FieldTreeModel::NAME).data(Qt::CheckStateRole).toInt(),
		    int(Qt::Checked));
		// A profile Squey ships can only be saved as another.
		PV_VALID(part<QPushButton>(dialog, "revert_profile_button")->isVisible(), true);
		PV_VALID(part<QPushButton>(dialog, "save_profile_button")->isVisible(), false);

		std::cout << "the preview" << std::endl;
		wait_for("the preview of http.host",
		         [&]() { return preview_column(dialog, "http.host").size() == 43; });
		const QStringList hosts = preview_column(dialog, "http.host");
		PV_VALID(hosts.count("www.ethereal.com"), qsizetype(1));
		PV_VALID(hosts.count(""), qsizetype(41));

		std::cout << "a wrong filter, then a right one" << std::endl;
		QLineEdit* filter = part<QLineEdit>(dialog, "filter_edit");
		QPushButton* import = part<QPushButton>(dialog, "process_import_button");
		filter->setText("http.reqest");
		wait_for("the check of the filter",
		         [&]() { return text_of(dialog, "filter_status").contains("not a valid"); });
		PV_VALID(import->isEnabled(), false);
		PV_VALID(text_of(dialog, "filter_status").contains("Show where"), true);
		filter->setText("http.request");
		wait_for("the check of the filter",
		         [&]() { return text_of(dialog, "filter_status").startsWith("✓"); });
		PV_VALID(import->isEnabled(), true);
		wait_for("the filtered preview",
		         [&]() { return preview_column(dialog, "http.host").size() == 2; });

		std::cout << "the other options, in their tab" << std::endl;
		QTabBar* tabs = part<QTabWidget>(dialog, "profile_tabs")->tabBar();
		const QComboBox* occurrence = part<QComboBox>(dialog, "occurrence_combo");
		PV_VALID(occurrence->isVisible(), false);
		QTest::mouseClick(tabs, Qt::LeftButton, {}, tabs->tabRect(1).center());
		PV_VALID(occurrence->isVisible(), true);
		// The filter stays at hand.
		PV_VALID(filter->isVisible(), true);
		QTest::mouseClick(tabs, Qt::LeftButton, {}, tabs->tabRect(0).center());
		PV_VALID(occurrence->isVisible(), false);

		std::cout << "saved as a profile of the user" << std::endl;
		part<QToolButton>(dialog, "duplicate_profile_button")->click();
		QListWidget* profiles = part<QListWidget>(dialog, "profiles_list");
		PV_VALID(profiles->currentItem()->text().toStdString(), std::string("HTTP copy"));
		// Named at once, in place.
		QLineEdit* naming = profiles->findChild<QLineEdit*>();
		PV_ASSERT_VALID(naming != nullptr, "the name of the copy", "can't be edited");
		naming->selectAll();
		QTest::keyClicks(naming, "Web");
		QTest::keyClick(naming, Qt::Key_Return);
		wait_for("the renaming", [&]() { return profiles->currentItem()->text() == "Web"; });
		PV_VALID(text_of(dialog, "profile_name").toStdString(), std::string("Web"));
		PV_VALID(QFile::exists(QString::fromStdString(pvpcap::get_user_profile_path("Web"))), true);
		const pvpcap::profile saved = pvpcap::read_profile(pvpcap::get_user_profile_path("Web"));
		PV_VALID(saved.columns.size(), size_t(28));
		PV_VALID(saved.filter, std::string("http.request"));
		// The changes went with the copy.
		choose_profile(dialog, "HTTP");
		PV_VALID(columns(dialog).size(), qsizetype(27));
		choose_profile(dialog, "Web");

		std::cout << "a change kept as a draft" << std::endl;
		QListWidget* list = part<QListWidget>(dialog, "columns_list");
		list->setCurrentRow(0);
		QTest::keyClick(list, Qt::Key_Delete);
		PV_VALID(columns(dialog).size(), qsizetype(27));
		dialog.reject();
	}

	std::cout << "opened again" << std::endl;
	{
		PVPcapsicum::ImportDialog dialog({});
		dialog.show();
		PV_VALID(text_of(dialog, "profile_name").toStdString(), std::string("Web"));
		PV_VALID(columns(dialog).size(), qsizetype(27));
		PV_VALID(text_of(dialog, "profile_state").contains("not saved"), true);
		PV_VALID(part<QPushButton>(dialog, "process_import_button")->isEnabled(), false);
		part<QPushButton>(dialog, "revert_profile_button")->click();
		PV_VALID(columns(dialog).size(), qsizetype(28));
		PV_VALID(text_of(dialog, "profile_state").isEmpty(), true);

		std::cout << "an option changed, then reverted" << std::endl;
		auto* tabs = part<QTabWidget>(dialog, "profile_tabs");
		PV_VALID(tabs->currentIndex(), 0);
		tabs->setCurrentIndex(1);
		QComboBox* occurrence = part<QComboBox>(dialog, "occurrence_combo");
		const int kept = occurrence->currentIndex();
		occurrence->setCurrentIndex((kept + 1) % occurrence->count());
		PV_VALID(text_of(dialog, "profile_state").isEmpty(), false);
		part<QPushButton>(dialog, "revert_profile_button")->click();
		PV_VALID(occurrence->currentIndex(), kept);
		PV_VALID(text_of(dialog, "profile_state").isEmpty(), true);
		dialog.reject();
	}

	std::cout << "a profile of an earlier version" << std::endl;
	{
		QFile legacy(QString::fromStdString(pvpcap::get_user_profile_path("Old")));
		PV_VALID(legacy.open(QIODevice::WriteOnly), true);
		legacy.write(R"({"options": {"info": true, "filters": "", "wireshark_profile": "Nowhere"},
		  "children": [{"fields": [{"filter_name": "ip.src", "type": "FT_IPv4",
		                            "select": true}], "children": []}]})");
		legacy.close();

		PVPcapsicum::ImportDialog dialog({capture});
		dialog.show();
		choose_profile(dialog, "Old");
		PV_VALID(columns(dialog).join(" ").toStdString(), std::string("ip.src _ws.col.Info"));
		// A Wireshark profile this machine lacks shows from the tab of the columns.
		PV_VALID(part<QTabWidget>(dialog, "profile_tabs")->tabIcon(1).isNull(), false);
		PV_VALID(text_of(dialog, "wireshark_problem").contains("Nowhere"), true);

		std::cout << "imported" << std::endl;
		// Once the capture was read: it could be one tshark can't read.
		wait_for("the first frames",
		         [&]() { return text_of(dialog, "fields_status").contains("first 43 frames"); });
		part<QPushButton>(dialog, "process_import_button")->click();
		PV_VALID(dialog.result(), int(QDialog::Accepted));
		PV_VALID(dialog.captures().join("|").toStdString(), capture.toStdString());
		PV_VALID(dialog.profile().columns.size(), size_t(2));
		PV_VALID(dialog.format().elementsByTagName("axis").size(), 4);
		// Not changed: no draft to keep.
		PV_VALID(pvpcap::read_drafts().empty(), true);
	}
	return 0;
}
