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

#include "CapturesPanel.h"

#include "Style.h"

#include <libpvpcap/extraction.h>

#include <pvkernel/core/PVFlatpakHelper.h>
#include <pvkernel/core/PVConfig.h>
#include <pvkernel/widgets/PVFileDialog.h>
#include <pvkernel/widgets/PVImportWidget.h>
#include <pvkernel/widgets/PVModdedIcon.h>

#include <QAction>
#include <QDirIterator>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QListWidget>
#include <QMenu>
#include <QPainter>
#include <QStandardPaths>
#include <QStyledItemDelegate>
#include <QToolButton>
#include <QVBoxLayout>

namespace PVPcapsicum
{

namespace
{

//! Read to estimate how many frames a capture has, which takes a few milliseconds.
constexpr uint64_t OVERVIEW_FRAMES = 20'000;
//! Read by tshark to tell the fields of a capture.
constexpr uint64_t SAMPLED_FRAMES = 2'000;
//! Of the first captures, whose fields are told: the others most likely hold the same.
constexpr int SAMPLED_CAPTURES = 4;

const QString LAST_DIRECTORY_KEY = "pcap/last_directory";

const QStringList CAPTURE_PATTERNS = {"*.pcap",     "*.pcapng",    "*.cap",      "*.dmp",
                                      "*.pcap.gz",  "*.pcapng.gz", "*.pcap.zst", "*.pcapng.zst",
                                      "*.pcap.xz",  "*.pcapng.xz", "*.pcap.bz2", "*.pcapng.bz2",
                                      "*.pcap.lz4", "*.pcapng.lz4"};

enum role { path_role = Qt::UserRole, details_role, size_role, state_role };
enum class state { known, reading, failed };

struct reading {
	std::optional<pvpcap::capture_overview> overview;
	std::optional<pvpcap::capture_sample> sample;
	QString error;
};

//! The name of a capture and its size, then what is known of it.
class CaptureDelegate : public QStyledItemDelegate
{
  public:
	using QStyledItemDelegate::QStyledItemDelegate;

	QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex&) const override
	{
		return {option.rect.width(), option.fontMetrics.height() * 2 + 14};
	}

	void paint(QPainter* painter,
	           const QStyleOptionViewItem& option,
	           const QModelIndex& index) const override
	{
		QStyleOptionViewItem background = option;
		initStyleOption(&background, index);
		background.text.clear();
		background.icon = QIcon();
		option.widget->style()->drawControl(QStyle::CE_ItemViewItem, &background, painter,
		                                    option.widget);

		painter->save();
		const QRect area = option.rect.adjusted(10, 6, -10, -6);
		const int line = option.fontMetrics.height();
		const QRect first(area.left(), area.top(), area.width(), line);
		const QRect second(area.left(), area.top() + line + 2, area.width(), line);
		const QColor secondary = style::secondary();
		const bool selected = option.state & QStyle::State_Selected;

		const QString size = index.data(size_role).toString();
		QFont small = option.font;
		small.setPointSizeF(small.pointSizeF() * 0.9);
		painter->setFont(small);
		painter->setPen(selected ? option.palette.color(QPalette::HighlightedText) : secondary);
		const int size_width = QFontMetrics(small).horizontalAdvance(size);
		painter->drawText(first, Qt::AlignRight | Qt::AlignVCenter, size);

		QFont bold = option.font;
		bold.setBold(true);
		painter->setFont(bold);
		painter->setPen(
		    option.palette.color(selected ? QPalette::HighlightedText : QPalette::Text));
		const QRect name_area = first.adjusted(0, 0, -size_width - 12, 0);
		painter->drawText(name_area, Qt::AlignLeft | Qt::AlignVCenter,
		                  QFontMetrics(bold).elidedText(index.data(Qt::DisplayRole).toString(),
		                                                Qt::ElideMiddle, name_area.width()));

		painter->setFont(small);
		const auto current = state(index.data(state_role).toInt());
		painter->setPen(current == state::failed ? style::error()
		                : selected               ? option.palette.color(QPalette::HighlightedText)
		                                         : secondary);
		painter->drawText(second, Qt::AlignLeft | Qt::AlignVCenter,
		                  QFontMetrics(small).elidedText(index.data(details_role).toString(),
		                                                 Qt::ElideRight, second.width()));
		painter->restore();
	}
};

//! Removes what is selected with the Delete key.
class CaptureList : public QListWidget
{
  public:
	explicit CaptureList(std::function<void()> remove_selected)
	    : _remove_selected(std::move(remove_selected))
	{
	}

  protected:
	void keyPressEvent(QKeyEvent* event) override
	{
		if (event->matches(QKeySequence::Delete) or event->key() == Qt::Key_Backspace) {
			_remove_selected();
			return;
		}
		QListWidget::keyPressEvent(event);
	}

  private:
	std::function<void()> _remove_selected;
};

QString details(const capture_info& info)
{
	if (not info.error.isEmpty()) {
		return info.error;
	}
	if (not info.overview) {
		return QObject::tr("Reading…");
	}
	const pvpcap::capture_overview& o = *info.overview;
	QStringList parts;
	if (o.format.empty()) {
		parts << QObject::tr("read by tshark only");
	} else {
		parts << QString::fromStdString(o.format) +
		             (o.compressed ? QObject::tr(", compressed") : QString());
	}
	if (o.whole) {
		parts << QObject::tr("%1 packets").arg(style::count_text(o.frames));
	} else if (o.estimated_frames != 0) {
		parts << QObject::tr("≈ %1 packets").arg(style::short_count_text(o.estimated_frames));
	}
	if (o.first_timestamp != 0) {
		parts << QObject::tr("from %1").arg(style::time_text(o.first_timestamp));
	}
	if (not o.damage.empty()) {
		parts << QObject::tr("cut short");
	}
	return parts.join(" · ");
}

} // namespace

CapturesPanel::CapturesPanel(QWidget* parent) : QWidget(parent)
{
	setAttribute(Qt::WA_StyledBackground);
	auto* layout = new QVBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(6);

	auto* header = new QHBoxLayout;
	header->addWidget(PVWidgets::PVImportWidget::section_title(tr("Captures")));
	header->addStretch();
	auto* add = new QToolButton;
	add->setObjectName("add_captures_button");
	add->setIcon(PVModdedIcon("folder-plus"));
	add->setToolTip(tr("Add captures (Ctrl+O)"));
	add->setAutoRaise(true);
	header->addWidget(add);
	_remove = new QToolButton;
	_remove->setObjectName("remove_captures_button");
	_remove->setIcon(PVModdedIcon("trash-xmark"));
	_remove->setToolTip(tr("Remove the selected captures (Delete)"));
	_remove->setAutoRaise(true);
	_remove->setEnabled(false);
	header->addWidget(_remove);
	layout->addLayout(header);

	// In the place of the list while it is empty.
	_ask = PVWidgets::PVImportWidget::drop_zone(PVModdedIcon("pcap"),
	                                            tr("Drop captures here,\nor click to choose them"));
	_ask->setToolTip(tr("Add captures (Ctrl+O)"));
	_ask->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	layout->addWidget(_ask);

	_list = new CaptureList([this]() { remove_selected(); });
	_list->setObjectName("captures_list");
	_list->setItemDelegate(new CaptureDelegate(_list));
	_list->setSelectionMode(QAbstractItemView::ExtendedSelection);
	// Moved up and down, which changes the numbers of their frames; files dropped on it go to
	// the dialog, which adds them.
	_list->setDragDropMode(QAbstractItemView::InternalMove);
	_list->setDefaultDropAction(Qt::MoveAction);
	_list->setContextMenuPolicy(Qt::CustomContextMenu);
	_list->setToolTip(tr("The frames of the captures are numbered in this order: drag them to "
	                     "change it."));
	layout->addWidget(_list);

	connect(add, &QToolButton::clicked, this, &CapturesPanel::ask);
	connect(_ask, &QToolButton::clicked, this, &CapturesPanel::ask);
	// One by one rather than from a braced list: MinGW stores the member pointers of such a list
	// as constants that point at the import thunks rather than at the signals of Qt, which
	// connect() then does not find.
	connect(_list->model(), &QAbstractItemModel::rowsInserted, this, &CapturesPanel::update_empty);
	connect(_list->model(), &QAbstractItemModel::rowsRemoved, this, &CapturesPanel::update_empty);
	update_empty();
	connect(_remove, &QToolButton::clicked, this, &CapturesPanel::remove_selected);
	connect(_list, &QListWidget::itemSelectionChanged, this,
	        [this]() { _remove->setEnabled(not _list->selectedItems().isEmpty()); });
	connect(_list->model(), &QAbstractItemModel::rowsMoved, this, &CapturesPanel::captures_moved);
	connect(_list, &QWidget::customContextMenuRequested, this, [this](const QPoint& at) {
		if (_list->selectedItems().isEmpty()) {
			return;
		}
		QMenu menu;
		menu.addAction(PVModdedIcon("trash-xmark"), tr("Remove"), this,
		               &CapturesPanel::remove_selected);
		menu.exec(_list->viewport()->mapToGlobal(at));
	});
}

CapturesPanel::~CapturesPanel() = default;

void CapturesPanel::update_empty()
{
	_ask->setVisible(_list->count() == 0);
	_list->setVisible(_list->count() != 0);
}

void CapturesPanel::ask()
{
	QString directory = PVCore::PVConfig::value(LAST_DIRECTORY_KEY).toString();
	// The Flatpak remembers directories of the host, which it does not see.
	if (directory.isEmpty() or
	    (not PVCore::PVFlatpakHelper::is_flatpak() and not QFileInfo(directory).isDir())) {
		directory = QStandardPaths::writableLocation(QStandardPaths::HomeLocation);
	}
	// tshark reads compressed captures, and so does the import.
	const QStringList files = PVWidgets::PVFileDialog::getOpenFileNames(
	    this, tr("Add captures"), directory,
	    tr("Captures (%1)").arg(CAPTURE_PATTERNS.join(' ')) + ";;" + tr("All files (*)"));
	if (not files.isEmpty()) {
		PVCore::PVConfig::set_value(LAST_DIRECTORY_KEY,
		                            PVCore::PVFlatpakHelper::host_directory_of(files.front()));
		add(files);
	}
}

void CapturesPanel::add(const QStringList& paths)
{
	QStringList files;
	for (const QString& path : paths) {
		const QFileInfo info(path);
		if (info.isDir()) {
			QStringList found;
			QDirIterator it(path, CAPTURE_PATTERNS, QDir::Files | QDir::Readable,
			                QDirIterator::Subdirectories);
			while (it.hasNext()) {
				found << it.next();
			}
			found.sort();
			files << found;
		} else if (info.isFile()) {
			files << info.absoluteFilePath();
		}
	}

	bool added = false;
	for (const QString& file : files) {
		const QString path = QFileInfo(file).absoluteFilePath();
		if (_entries.contains(path)) {
			continue;
		}
		entry& e = _entries[path];
		e.info.path = path;
		e.job = std::make_unique<BackgroundJob>();
		auto* item = new QListWidgetItem(QFileInfo(path).fileName(), _list);
		item->setData(path_role, path);
		item->setData(size_role, style::size_text(uint64_t(QFileInfo(path).size())));
		item->setToolTip(path);
		read(path, true);
		added = true;
	}
	if (added) {
		Q_EMIT captures_changed();
		Q_EMIT infos_changed();
	}
}

QStringList CapturesPanel::captures() const
{
	QStringList paths;
	for (int i = 0; i < _list->count(); i++) {
		paths << _list->item(i)->data(path_role).toString();
	}
	return paths;
}

std::vector<const capture_info*> CapturesPanel::infos() const
{
	std::vector<const capture_info*> infos;
	for (const QString& path : captures()) {
		infos.push_back(&_entries.at(path).info);
	}
	return infos;
}

void CapturesPanel::set_reading_options(const std::vector<std::string>& options)
{
	if (options == _reading_options) {
		return;
	}
	_reading_options = options;
	const QStringList paths = captures();
	for (int i = 0; i < std::min<int>(paths.size(), SAMPLED_CAPTURES); i++) {
		read(paths[i], false);
	}
}

void CapturesPanel::read(const QString& path, bool overview)
{
	entry& e = _entries.at(path);
	const bool sampled = captures().indexOf(path) < SAMPLED_CAPTURES;
	if (not overview and not sampled) {
		return;
	}
	e.info.reading = true;
	const std::vector<std::string> options = _reading_options;
	e.job->run(
	    [path, options, overview, sampled](const std::atomic<bool>& canceled) {
		    reading r;
		    const std::string file = path.toStdString();
		    if (overview) {
			    r.overview = pvpcap::overview_capture(file, OVERVIEW_FRAMES);
		    }
		    if (sampled and not canceled) {
			    try {
				    r.sample = pvpcap::sample_capture(file, SAMPLED_FRAMES, options, canceled);
			    } catch (const std::exception& error) {
				    r.error = QString::fromStdString(error.what());
				    if (r.error.startsWith("tshark: ")) {
					    r.error.remove(0, 8);
				    }
			    }
		    }
		    return r;
	    },
	    [this, path, sampled](reading r) {
		    capture_info& info = _entries.at(path).info;
		    if (r.overview) {
			    info.overview = std::move(r.overview);
		    }
		    if (sampled) {
			    info.sample = std::move(r.sample);
			    info.error = r.error;
		    }
		    info.reading = false;
		    update_item(path);
		    Q_EMIT infos_changed();
		    if (sampled) {
			    Q_EMIT samples_changed();
		    }
	    });
	update_item(path);
}

void CapturesPanel::update_item(const QString& path)
{
	const capture_info& info = _entries.at(path).info;
	for (int i = 0; i < _list->count(); i++) {
		QListWidgetItem* item = _list->item(i);
		if (item->data(path_role).toString() != path) {
			continue;
		}
		const QString text = details(info);
		item->setData(details_role, text);
		item->setData(state_role, int(not info.error.isEmpty() ? state::failed
		                              : info.reading           ? state::reading
		                                                       : state::known));
		item->setToolTip(path + "\n" + text);
	}
}

void CapturesPanel::remove_selected()
{
	const QList<QListWidgetItem*> selected = _list->selectedItems();
	if (selected.isEmpty()) {
		return;
	}
	for (QListWidgetItem* item : selected) {
		_entries.erase(item->data(path_role).toString());
		delete item;
	}
	captures_moved();
}

void CapturesPanel::captures_moved()
{
	// The first captures are the ones whose fields are told.
	const QStringList paths = captures();
	for (int i = 0; i < std::min<int>(paths.size(), SAMPLED_CAPTURES); i++) {
		const capture_info& info = _entries.at(paths[i]).info;
		if (not info.sample and not info.reading and info.error.isEmpty()) {
			read(paths[i], false);
		}
	}
	Q_EMIT captures_changed();
	Q_EMIT infos_changed();
	Q_EMIT samples_changed();
}

} // namespace PVPcapsicum
