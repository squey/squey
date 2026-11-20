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

#include <pvkernel/widgets/PVFileListImportWidget.h>

#include <pvkernel/core/PVTaskProgress.h>
#include <pvkernel/rush/PVFileListDescription.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVInputTypeFileList.h>
#include <pvkernel/widgets/PVFileDialog.h>
#include <pvkernel/widgets/PVImportFilterWidget.h>

#include <QFileInfo>
#include <QHBoxLayout>
#include <QHash>
#include <QLabel>
#include <QListWidget>
#include <QPushButton>
#include <QRadioButton>
#include <QStandardPaths>
#include <QVBoxLayout>

#include <optional>

PVWidgets::PVFileListImportWidget::PVFileListImportWidget(const PVRush::PVInputTypeFileList& type,
                                                          QWidget* parent)
    : PVImportWidget(parent)
    , _type(type)
    , _files(new QListWidget)
    , _remove(new QPushButton(tr("Remove")))
    , _sources(new QWidget)
    , _source_per_file(new QRadioButton(tr("One source for each file")))
    , _filter(new PVImportFilterWidget)
{
	auto* add = new QPushButton(tr("Add files..."));
	_files->setObjectName("import_files");
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
	_filter->hide();

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
	layout->addWidget(_filter);

	connect(add, &QPushButton::clicked, this, &PVFileListImportWidget::add_files);
	connect(_remove, &QPushButton::clicked, this, &PVFileListImportWidget::remove_files);
	connect(_files, &QListWidget::itemSelectionChanged, this,
	        [this]() { _remove->setEnabled(not _files->selectedItems().isEmpty()); });
	// The condition reads the columns of the files of a source, of which only those of several
	// files name them.
	connect(_source_per_file, &QRadioButton::toggled, _filter,
	        &PVImportFilterWidget::check_row_filter);
	connect(_filter, &PVImportFilterWidget::changed, this, &PVImportWidget::ready_changed);
}

bool PVWidgets::PVFileListImportWidget::is_ready() const
{
	return _description != nullptr and _filter->is_valid();
}

std::vector<PVWidgets::PVImportWidget::source_t> PVWidgets::PVFileListImportWidget::sources() const
{
	const auto source_of = [this](QStringList const& paths) -> source_t {
		const QList<PVRush::PVInputDescription_p> inputs{description_of(paths)};
		return {inputs, _type.get_format_from_inputs(inputs)};
	};
	std::vector<source_t> sources;
	if (one_source_per_file()) {
		for (const QString& path : _paths) {
			sources.push_back(source_of({path}));
		}
	} else {
		sources.push_back(source_of(_paths));
	}
	return sources;
}

void PVWidgets::PVFileListImportWidget::set_paths(QStringList const& paths)
{
	_paths = paths;
	_files->clear();
	for (const QString& path : _paths) {
		auto* item = new QListWidgetItem(QFileInfo(path).fileName(), _files);
		item->setToolTip(path);
	}
	_sources->setVisible(_paths.size() > 1);
	describe();
}

void PVWidgets::PVFileListImportWidget::add_files()
{
	const QStringList chosen = PVFileDialog::getOpenFileNames(
	    this, _type.files_dialog_title(),
	    _paths.isEmpty() ? QStandardPaths::writableLocation(QStandardPaths::HomeLocation)
	                     : QFileInfo(_paths.back()).absolutePath(),
	    _type.files_dialog_filter());
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

void PVWidgets::PVFileListImportWidget::remove_files()
{
	QStringList kept;
	for (int row = 0; row < _files->count(); row++) {
		if (not _files->item(row)->isSelected()) {
			kept << _paths[row];
		}
	}
	set_paths(kept);
}

void PVWidgets::PVFileListImportWidget::describe()
{
	_description.reset();
	_filter->hide();
	Q_EMIT ready_changed();
	if (_paths.isEmpty()) {
		return;
	}

	struct read_t {
		std::shared_ptr<PVRush::PVFileListDescription> description;
		QStringList warnings;
		QHash<QString, double> fill_rates;
		PVRush::PVFormat format;
	};
	auto read = std::make_shared<read_t>();
	run_task(
	    tr("Reading the files"),
	    [type = &_type, paths = _paths, read](PVCore::PVTaskProgress& progress) {
		    read->description = type->describe(paths, progress, read->warnings);
		    if (read->description != nullptr) {
			    read->fill_rates = type->fill_rates(*read->description);
			    read->format = type->get_format_from_inputs({read->description});
		    }
	    },
	    [this, read](bool completed) {
		    if (not completed or read->description == nullptr) {
			    return;
		    }
		    _description = read->description;
		    for (const QString& warning : read->warnings) {
			    Q_EMIT message(Severity::warning, warning.toHtmlEscaped());
		    }
		    std::vector<PVImportFilterWidget::Column> columns;
		    const QList<PVRush::PVAxisFormat>& axes = read->format.get_axes();
		    // But for the column of the names of the files, which a source of several files has.
		    for (qsizetype axis = _description->multi_inputs() ? 1 : 0; axis < axes.size();
		         axis++) {
			    const QString& name = axes[axis].get_name();
			    columns.push_back({name, axes[axis].get_type(),
			                       read->fill_rates.contains(name)
			                           ? std::optional<double>(read->fill_rates[name])
			                           : std::nullopt});
		    }
		    _filter->set_columns(columns, {}, {}, [this](QString const& condition) {
			    const QStringList paths =
			        one_source_per_file() ? QStringList{_paths.front()} : _paths;
			    return _type.row_filter_error(*_type.with_paths(*_description, paths), condition);
		    });
		    _filter->show();
		    Q_EMIT ready_changed();
	    });
}

bool PVWidgets::PVFileListImportWidget::one_source_per_file() const
{
	return _paths.size() > 1 and _source_per_file->isChecked();
}

std::shared_ptr<PVRush::PVFileListDescription>
PVWidgets::PVFileListImportWidget::description_of(QStringList const& paths) const
{
	std::shared_ptr<PVRush::PVFileListDescription> description =
	    _type.with_paths(*_description, paths);
	description->set_columns(_filter->columns());
	description->set_row_filter(_filter->row_filter());
	return description;
}
