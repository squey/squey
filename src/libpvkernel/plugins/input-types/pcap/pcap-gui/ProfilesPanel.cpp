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

#include "ProfilesPanel.h"

#include "ProfileEditor.h"
#include "Style.h"

#include <pvkernel/core/PVConfig.h>
#include <pvkernel/widgets/PVFileDialog.h>
#include <pvkernel/widgets/PVModdedIcon.h>

#include <QFileInfo>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QListWidget>
#include <QMenu>
#include <QMessageBox>
#include <QPainter>
#include <QStandardPaths>
#include <QStyledItemDelegate>
#include <QToolButton>
#include <QVBoxLayout>

namespace PVPcapsicum
{

namespace
{

enum role { builtin_role = Qt::UserRole, details_role, modified_role, problem_role };

const QString LAST_PROFILE_DIRECTORY_KEY = "pcap/last_profile_directory";

//! The name of a profile, what it is, then what it imports.
class ProfileDelegate : public QStyledItemDelegate
{
  public:
	using QStyledItemDelegate::QStyledItemDelegate;

	QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex&) const override
	{
		return {option.rect.width(), option.fontMetrics.height() * 2 + 12};
	}

	void updateEditorGeometry(QWidget* editor,
	                          const QStyleOptionViewItem& option,
	                          const QModelIndex&) const override
	{
		editor->setGeometry(option.rect.adjusted(6, 3, -6, -option.rect.height() / 2 + 2));
	}

	void paint(QPainter* painter,
	           const QStyleOptionViewItem& option,
	           const QModelIndex& index) const override
	{
		QStyleOptionViewItem background = option;
		initStyleOption(&background, index);
		background.text.clear();
		option.widget->style()->drawControl(QStyle::CE_ItemViewItem, &background, painter,
		                                    option.widget);

		painter->save();
		const bool selected = option.state & QStyle::State_Selected;
		const QColor secondary =
		    selected ? option.palette.color(QPalette::HighlightedText) : style::secondary();
		const QRect area = option.rect.adjusted(10, 6, -10, -6);
		const int line = option.fontMetrics.height();
		const QRect first(area.left(), area.top(), area.width(), line);
		const QRect second(area.left(), area.top() + line + 2, area.width(), line);

		QFont bold = option.font;
		bold.setBold(true);
		painter->setFont(bold);
		painter->setPen(
		    option.palette.color(selected ? QPalette::HighlightedText : QPalette::Text));
		const QString name = QFontMetrics(bold).elidedText(index.data(Qt::DisplayRole).toString(),
		                                                   Qt::ElideRight, area.width() - 110);
		painter->drawText(first, Qt::AlignLeft | Qt::AlignVCenter, name);
		int x = first.left() + QFontMetrics(bold).horizontalAdvance(name) + 8;
		if (index.data(builtin_role).toBool()) {
			x += style::paint_tag(*painter, x, first, QObject::tr("built-in"), secondary) + 4;
		}
		if (not index.data(problem_role).toString().isEmpty()) {
			x +=
			    style::paint_tag(*painter, x, first, QObject::tr("unreadable"), style::error()) + 4;
		}
		if (index.data(modified_role).toBool()) {
			style::paint_tag(*painter, x, first, QObject::tr("modified"), style::accent());
		}

		QFont small = option.font;
		small.setPointSizeF(small.pointSizeF() * 0.9);
		painter->setFont(small);
		painter->setPen(secondary);
		painter->drawText(second, Qt::AlignLeft | Qt::AlignVCenter,
		                  QFontMetrics(small).elidedText(index.data(details_role).toString(),
		                                                 Qt::ElideRight, second.width()));
		painter->restore();
	}
};

QString details(const pvpcap::profile& p)
{
	QString text = p.columns.size() == 1 ? QObject::tr("1 column")
	                                     : QObject::tr("%1 columns").arg(p.columns.size());
	if (not p.filter.empty()) {
		text += " · " + QObject::tr("frames of %1").arg(QString::fromStdString(p.filter));
	}
	return text;
}

} // namespace

ProfilesPanel::ProfilesPanel(ProfileEditor* editor, QWidget* parent)
    : QWidget(parent), _editor(editor), _menu(new QMenu(this))
{
	setAttribute(Qt::WA_StyledBackground);
	auto* layout = new QVBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(6);

	QAction* create = _menu->addAction(tr("&New profile…"), this, &ProfilesPanel::create);
	QAction* copy =
	    _menu->addAction(PVModdedIcon("copy"), tr("&Duplicate"), this, &ProfilesPanel::duplicate);
	_rename = _menu->addAction(PVModdedIcon("pen-to-square"), tr("&Rename"), this,
	                           &ProfilesPanel::rename);
	_rename->setShortcut(Qt::Key_F2);
	_remove =
	    _menu->addAction(PVModdedIcon("trash-xmark"), tr("De&lete…"), this, &ProfilesPanel::remove);
	_menu->addSeparator();
	_menu->addAction(PVModdedIcon("file-import"), tr("&Import from a file…"), this,
	                 &ProfilesPanel::import);
	_menu->addAction(PVModdedIcon("file-export"), tr("E&xport to a file…"), this,
	                 &ProfilesPanel::export_profile);

	auto* header = new QHBoxLayout;
	header->addWidget(style::section_title(tr("Profiles")));
	header->addStretch();
	const auto tool = [&](QAction* action, const char* name) {
		auto* button = new QToolButton;
		button->setObjectName(name);
		button->setDefaultAction(action);
		button->setAutoRaise(true);
		header->addWidget(button);
		return button;
	};
	QToolButton* create_button = tool(create, "new_profile_button");
	create_button->setText("+");
	create_button->setToolTip(tr("New profile"));
	tool(copy, "duplicate_profile_button")->setToolTip(tr("Duplicate the profile, as changed"));
	auto* more = new QToolButton;
	more->setObjectName("profile_actions_button");
	more->setIcon(PVModdedIcon("bars"));
	more->setToolTip(tr("More actions on profiles"));
	more->setAutoRaise(true);
	more->setPopupMode(QToolButton::InstantPopup);
	more->setMenu(_menu);
	header->addWidget(more);
	layout->addLayout(header);

	_list = new QListWidget;
	_list->setObjectName("profiles_list");
	_list->setItemDelegate(new ProfileDelegate(_list));
	_list->setEditTriggers(QAbstractItemView::EditKeyPressed);
	_list->setContextMenuPolicy(Qt::CustomContextMenu);
	_list->addAction(_rename);
	layout->addWidget(_list, 1);

	connect(_list, &QWidget::customContextMenuRequested, this,
	        [this](const QPoint& at) { _menu->exec(_list->viewport()->mapToGlobal(at)); });
	connect(_list, &QListWidget::currentRowChanged, this, [this](int row) {
		if (not _refreshing and row >= 0 and row < int(_editor->profiles().size())) {
			_editor->choose(_editor->profiles()[row]);
		}
	});
	connect(_list, &QListWidget::itemChanged, this, &ProfilesPanel::renamed);
	connect(_editor, &ProfileEditor::profiles_changed, this, &ProfilesPanel::refresh);
	connect(_editor, &ProfileEditor::current_changed, this, &ProfilesPanel::select_current);
	connect(_editor, &ProfileEditor::profile_changed, this, &ProfilesPanel::refresh);
	refresh();
}

void ProfilesPanel::refresh()
{
	_refreshing = true;
	const auto& profiles = _editor->profiles();
	while (_list->count() > int(profiles.size())) {
		delete _list->takeItem(_list->count() - 1);
	}
	for (size_t i = 0; i < profiles.size(); i++) {
		QListWidgetItem* item =
		    int(i) < _list->count() ? _list->item(int(i)) : new QListWidgetItem(_list);
		const pvpcap::profile_location& location = profiles[i];
		item->setText(QString::fromStdString(location.name));
		item->setData(builtin_role, location.builtin);
		item->setData(modified_role, _editor->modified(location));
		const QString problem = _editor->problem(location);
		item->setData(problem_role, problem);
		item->setData(details_role,
		              problem.isEmpty() ? details(_editor->profile(location)) : problem);
		item->setToolTip(location.builtin
		                     ? tr("Shipped with Squey: its changes can only be saved as a profile "
		                          "of yours.")
		                     : QString::fromStdString(location.path()));
		item->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable |
		               (location.builtin ? Qt::NoItemFlags : Qt::ItemIsEditable));
	}
	_refreshing = false;
	select_current();
}

void ProfilesPanel::select_current()
{
	const auto& profiles = _editor->profiles();
	const auto it = std::find(profiles.begin(), profiles.end(), _editor->current());
	_refreshing = true;
	if (it != profiles.end()) {
		_list->setCurrentRow(int(it - profiles.begin()));
	}
	_refreshing = false;
	_rename->setEnabled(not _editor->current().builtin);
	_remove->setEnabled(not _editor->current().builtin);
}

bool ProfilesPanel::attempt(const QString& what, const std::function<void()>& change)
{
	try {
		change();
		return true;
	} catch (const pvpcap::profile_error& e) {
		QMessageBox::warning(this, what, QString::fromStdString(e.what()));
		return false;
	}
}

void ProfilesPanel::create()
{
	bool ok = false;
	const QString name = QInputDialog::getText(
	    this, tr("New profile"), tr("Name of the new profile:"), QLineEdit::Normal,
	    QString::fromStdString(pvpcap::free_profile_name(tr("New profile").toStdString())), &ok);
	if (ok) {
		attempt(tr("Can't create the profile"), [&]() { _editor->create(name.toStdString()); });
	}
}

void ProfilesPanel::duplicate()
{
	const std::string name = pvpcap::free_profile_name(_editor->current().name + " copy");
	if (attempt(tr("Can't duplicate the profile"), [&]() { _editor->save_as(name); })) {
		// To give it its name at once.
		_list->editItem(_list->currentItem());
	}
}

void ProfilesPanel::rename()
{
	if (not _editor->current().builtin and _list->currentItem() != nullptr) {
		_list->editItem(_list->currentItem());
	}
}

void ProfilesPanel::renamed(QListWidgetItem* item)
{
	if (_refreshing or _list->row(item) != _list->currentRow()) {
		return;
	}
	const std::string name = item->text().trimmed().toStdString();
	if (name == _editor->current().name) {
		return;
	}
	if (not attempt(tr("Can't rename the profile"), [&]() { _editor->rename(name); })) {
		refresh();
	}
}

void ProfilesPanel::remove()
{
	const pvpcap::profile_location location = _editor->current();
	if (location.builtin) {
		return;
	}
	if (QMessageBox::question(
	        this, tr("Delete the profile"),
	        tr("Delete the profile “%1”?").arg(QString::fromStdString(location.name))) !=
	    QMessageBox::Yes) {
		return;
	}
	attempt(tr("Can't delete the profile"), [&]() { _editor->remove(); });
}

void ProfilesPanel::import()
{
	QString directory = PVCore::PVConfig::value(LAST_PROFILE_DIRECTORY_KEY).toString();
	if (directory.isEmpty()) {
		directory = QStandardPaths::writableLocation(QStandardPaths::HomeLocation);
	}
	const QStringList files = PVWidgets::PVFileDialog::getOpenFileNames(
	    this, tr("Import profiles"), directory,
	    tr("Profiles (*.json)") + ";;" + tr("All files (*)"));
	for (const QString& file : files) {
		PVCore::PVConfig::set_value(LAST_PROFILE_DIRECTORY_KEY, QFileInfo(file).absolutePath());
		const std::string name =
		    pvpcap::free_profile_name(QFileInfo(file).completeBaseName().toStdString());
		attempt(tr("Can't import the profile"),
		        [&]() { _editor->import_file(file.toStdString(), name); });
	}
}

void ProfilesPanel::export_profile()
{
	QString directory = PVCore::PVConfig::value(LAST_PROFILE_DIRECTORY_KEY).toString();
	if (directory.isEmpty()) {
		directory = QStandardPaths::writableLocation(QStandardPaths::HomeLocation);
	}
	const QString file = PVWidgets::PVFileDialog::getSaveFileName(
	    this, tr("Export the profile"),
	    directory + "/" + QString::fromStdString(_editor->current().name) + ".json",
	    tr("Profiles (*.json)"));
	if (not file.isEmpty()) {
		PVCore::PVConfig::set_value(LAST_PROFILE_DIRECTORY_KEY, QFileInfo(file).absolutePath());
		attempt(tr("Can't export the profile"),
		        [&]() { _editor->export_file(file.toStdString()); });
	}
}

} // namespace PVPcapsicum
