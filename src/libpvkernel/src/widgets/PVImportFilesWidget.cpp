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

#include <pvkernel/widgets/PVFileDialog.h>

#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QPushButton>
#include <QRadioButton>
#include <QStandardPaths>
#include <QVBoxLayout>

#include <utility>

PVWidgets::PVImportFilesWidget::PVImportFilesWidget(QString dialog_title,
                                                    QString dialog_filter,
                                                    QWidget* parent)
    : QWidget(parent)
    , _dialog_title(std::move(dialog_title))
    , _dialog_filter(std::move(dialog_filter))
    , _files(new QListWidget)
    , _remove(new QPushButton(tr("Remove")))
    , _sources(new QWidget)
    , _source_per_file(new QRadioButton(tr("One source for each file")))
{
	auto* add = new QPushButton(tr("Add files..."));
	_files->setObjectName("import_files");
	_source_per_file->setObjectName("import_source_per_file");
	_files->setSelectionMode(QAbstractItemView::ExtendedSelection);
	_files->setMaximumHeight(_files->fontMetrics().lineSpacing() * 8);
	_remove->setEnabled(false);

	auto* one_source = new QRadioButton(tr("One source of all the files"));
	one_source->setChecked(true);
	auto* sources_layout = new QHBoxLayout(_sources);
	sources_layout->setContentsMargins(0, 0, 0, 0);
	sources_layout->addWidget(one_source);
	sources_layout->addWidget(_source_per_file);
	sources_layout->addStretch();
	_sources->hide();

	auto* file_buttons = new QVBoxLayout;
	file_buttons->addWidget(add);
	file_buttons->addWidget(_remove);
	file_buttons->addStretch();
	auto* files_layout = new QHBoxLayout;
	files_layout->addWidget(_files);
	files_layout->addLayout(file_buttons);

	auto* layout = new QVBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);
	layout->addWidget(new QLabel(tr("Files")));
	layout->addLayout(files_layout);
	layout->addWidget(_sources);

	connect(add, &QPushButton::clicked, this, &PVImportFilesWidget::add_files);
	connect(_remove, &QPushButton::clicked, this, &PVImportFilesWidget::remove_files);
	connect(_files, &QListWidget::itemSelectionChanged, this,
	        [this]() { _remove->setEnabled(not _files->selectedItems().isEmpty()); });
	connect(_source_per_file, &QRadioButton::toggled, this, &PVImportFilesWidget::sources_changed);
}

void PVWidgets::PVImportFilesWidget::set_paths(QStringList const& paths)
{
	_paths = paths;
	_files->clear();
	for (const QString& path : _paths) {
		auto* item = new QListWidgetItem(QFileInfo(path).fileName(), _files);
		item->setToolTip(path);
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

void PVWidgets::PVImportFilesWidget::add_files()
{
	const QStringList chosen = PVFileDialog::getOpenFileNames(
	    this, _dialog_title,
	    _paths.isEmpty() ? QStandardPaths::writableLocation(QStandardPaths::HomeLocation)
	                     : QFileInfo(_paths.back()).absolutePath(),
	    _dialog_filter);
	QStringList paths = _paths;
	for (const QString& path : chosen) {
		if (not paths.contains(path)) {
			paths << path;
		}
	}
	if (paths != _paths) {
		set_paths(paths);
	}
}

void PVWidgets::PVImportFilesWidget::remove_files()
{
	QStringList kept;
	for (int row = 0; row < _files->count(); row++) {
		if (not _files->item(row)->isSelected()) {
			kept << _paths[row];
		}
	}
	set_paths(kept);
}
