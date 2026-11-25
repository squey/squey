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

#include <pvkernel/widgets/PVImportFilesWidget.h>

#include <pvkernel/core/PVFlatpakHelper.h>
#include <pvkernel/widgets/PVFileDialog.h>
#include <pvkernel/widgets/PVImportWidget.h>
#include <pvkernel/widgets/PVModdedIcon.h>

#include <QAction>
#include <QDir>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLocale>
#include <QPushButton>
#include <QRadioButton>
#include <QStandardPaths>
#include <QToolButton>
#include <QTreeWidget>
#include <QVBoxLayout>

#include <algorithm>
#include <utility>

namespace
{

enum FileColumn { FILE_NAME, FILE_SIZE, FILE_FOLDER };

/** Such as "~/data" for a folder of the home directory. */
QString folder_of(QString const& path)
{
	const QString folder =
	    QDir::toNativeSeparators(PVCore::PVFlatpakHelper::host_directory_of(path));
	const QString home = QDir::toNativeSeparators(QDir::homePath());
	if (folder == home or folder.startsWith(home + QDir::separator())) {
		return "~" + folder.mid(home.size());
	}
	return folder;
}

/** Such as "54.2 MB", as the captures of pcap tell their size. */
QString size_of(qint64 bytes)
{
	return QLocale().formattedDataSize(bytes, 1, QLocale::DataSizeSIFormat);
}

QToolButton* file_tool_button(QString const& icon, QString const& tip)
{
	auto* button = new QToolButton;
	button->setIcon(PVModdedIcon(icon));
	button->setToolTip(tip);
	button->setAutoRaise(true);
	return button;
}

} // namespace

PVWidgets::PVImportFilesWidget::PVImportFilesWidget(QString dialog_title,
                                                    QString dialog_filter,
                                                    QIcon icon,
                                                    QWidget* parent)
    : QWidget(parent)
    , _dialog_title(std::move(dialog_title))
    , _dialog_filter(std::move(dialog_filter))
    , _icon(std::move(icon))
    , _ask(PVImportWidget::drop_zone(_icon, tr("Drop files here,\nor click to choose them")))
    , _files(new QTreeWidget)
    , _summary(new QLabel)
    , _remove(file_tool_button("trash-xmark", tr("Remove the selected files (Delete)")))
    , _sources(new QWidget)
    , _source_per_file(new QRadioButton(tr("One source for each file")))
{
	QToolButton* add = file_tool_button("folder-plus", tr("Add files"));
	add->setObjectName("import_add_files");
	_remove->setObjectName("import_remove_files");
	_remove->setEnabled(false);
	_summary->setObjectName("import_files_summary");
	_summary->setProperty("secondary", true);

	_files->setObjectName("import_files");
	_source_per_file->setObjectName("import_source_per_file");
	_files->setHeaderLabels({tr("Name"), tr("Size"), tr("Folder")});
	_files->setRootIsDecorated(false);
	_files->setUniformRowHeights(true);
	_files->setSelectionMode(QAbstractItemView::ExtendedSelection);
	_files->header()->setSectionResizeMode(FILE_NAME, QHeaderView::ResizeToContents);
	_files->header()->setSectionResizeMode(FILE_SIZE, QHeaderView::ResizeToContents);
	_files->hide();
	auto* remove = new QAction(PVModdedIcon("trash-xmark"), tr("Remove"), _files);
	remove->setShortcuts({QKeySequence::Delete, QKeySequence(Qt::Key_Backspace)});
	remove->setShortcutContext(Qt::WidgetShortcut);
	_files->addAction(remove);
	_files->setContextMenuPolicy(Qt::ActionsContextMenu);

	auto* one_source = new QRadioButton(tr("One source of all the files"));
	one_source->setChecked(true);
	auto* sources_layout = new QHBoxLayout(_sources);
	sources_layout->setContentsMargins(0, 0, 0, 0);
	sources_layout->addWidget(one_source);
	sources_layout->addWidget(_source_per_file);
	sources_layout->addStretch();
	_sources->hide();

	auto* header = new QHBoxLayout;
	header->addWidget(PVImportWidget::section_title(tr("Files")));
	header->addStretch();
	header->addWidget(_summary);
	header->addWidget(add);
	header->addWidget(_remove);

	auto* layout = new QVBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);
	layout->addLayout(header);
	layout->addWidget(_ask);
	layout->addWidget(_files);
	layout->addWidget(_sources);

	connect(add, &QToolButton::clicked, this, &PVImportFilesWidget::add_files);
	connect(_ask, &QToolButton::clicked, this, &PVImportFilesWidget::add_files);
	connect(_remove, &QToolButton::clicked, this, &PVImportFilesWidget::remove_files);
	connect(remove, &QAction::triggered, this, &PVImportFilesWidget::remove_files);
	connect(_files, &QTreeWidget::itemSelectionChanged, this,
	        [this]() { _remove->setEnabled(not _files->selectedItems().isEmpty()); });
	connect(_source_per_file, &QRadioButton::toggled, this, &PVImportFilesWidget::sources_changed);
}

void PVWidgets::PVImportFilesWidget::set_paths(QStringList const& paths)
{
	_paths = paths;
	_files->clear();
	qint64 size = 0;
	for (const QString& path : _paths) {
		const QFileInfo info(path);
		size += info.size();
		auto* item =
		    new QTreeWidgetItem(_files, {info.fileName(), size_of(info.size()), folder_of(path)});
		item->setIcon(FILE_NAME, _icon);
		item->setTextAlignment(FILE_SIZE, Qt::AlignRight | Qt::AlignVCenter);
		item->setToolTip(FILE_NAME,
		                 QDir::toNativeSeparators(PVCore::PVFlatpakHelper::host_path(path)));
		item->setToolTip(FILE_FOLDER, QDir::toNativeSeparators(info.absolutePath()));
	}
	// As tall as its rows, from three to ten, then it scrolls.
	if (not _paths.isEmpty()) {
		const int rows = std::clamp(int(_paths.size()), 3, 10);
		_files->setFixedHeight(_files->header()->sizeHint().height() +
		                       rows * _files->sizeHintForRow(0) + 2 * _files->frameWidth());
	}
	_ask->setVisible(_paths.isEmpty());
	_files->setVisible(not _paths.isEmpty());
	_remove->setEnabled(false);
	if (_paths.isEmpty()) {
		_summary->clear();
	} else {
		const QString total = size_of(size);
		_summary->setText(_paths.size() == 1 ? tr("1 file, %1").arg(total)
		                                     : tr("%1 files, %2").arg(_paths.size()).arg(total));
	}
	_sources->setVisible(_paths.size() > 1);
	Q_EMIT paths_changed();
}

bool PVWidgets::PVImportFilesWidget::one_source_per_file() const
{
	return _paths.size() > 1 and _source_per_file->isChecked();
}

std::vector<QStringList> PVWidgets::PVImportFilesWidget::sources() const
{
	if (not one_source_per_file()) {
		return {_paths};
	}
	std::vector<QStringList> sources;
	for (const QString& path : _paths) {
		sources.push_back({path});
	}
	return sources;
}

void PVWidgets::PVImportFilesWidget::add_paths(QStringList const& paths)
{
	QStringList added = _paths;
	for (const QString& path : paths) {
		if (not added.contains(path)) {
			added << path;
		}
	}
	if (added != _paths) {
		set_paths(added);
	}
}

void PVWidgets::PVImportFilesWidget::add_files()
{
	add_paths(PVFileDialog::getOpenFileNames(
	    this, _dialog_title,
	    _paths.isEmpty() ? QStandardPaths::writableLocation(QStandardPaths::HomeLocation)
	                     : PVCore::PVFlatpakHelper::host_directory_of(_paths.back()),
	    _dialog_filter));
}

void PVWidgets::PVImportFilesWidget::remove_files()
{
	QStringList kept;
	for (int row = 0; row < _files->topLevelItemCount(); row++) {
		if (not _files->topLevelItem(row)->isSelected()) {
			kept << _paths[row];
		}
	}
	set_paths(kept);
}
