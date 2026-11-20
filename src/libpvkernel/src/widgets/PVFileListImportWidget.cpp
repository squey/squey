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
#include <pvkernel/widgets/PVImportFilesWidget.h>
#include <pvkernel/widgets/PVImportFilterWidget.h>

#include <QHash>
#include <QVBoxLayout>

#include <optional>

PVWidgets::PVFileListImportWidget::PVFileListImportWidget(const PVRush::PVInputTypeFileList& type,
                                                          QWidget* parent)
    : PVImportWidget(parent)
    , _type(type)
    , _files(new PVImportFilesWidget(type.files_dialog_title(), type.files_dialog_filter()))
    , _filter(new PVImportFilterWidget)
{
	_filter->hide();

	auto* layout = new QVBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);
	layout->addWidget(_files);
	layout->addWidget(_filter);

	connect(_files, &PVImportFilesWidget::paths_changed, this, &PVFileListImportWidget::describe);
	// The condition reads the columns of the files of a source, of which only those of several
	// files name them.
	connect(_files, &PVImportFilesWidget::sources_changed, _filter,
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
	for (const QStringList& paths : _files->sources()) {
		sources.push_back(source_of(paths));
	}
	return sources;
}

void PVWidgets::PVFileListImportWidget::set_paths(QStringList const& paths)
{
	_files->set_paths(paths);
}

void PVWidgets::PVFileListImportWidget::add_paths(QStringList const& paths)
{
	_files->add_paths(paths);
}

void PVWidgets::PVFileListImportWidget::describe()
{
	_description.reset();
	_filter->hide();
	Q_EMIT ready_changed();
	if (_files->paths().isEmpty()) {
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
	    [type = &_type, paths = _files->paths(), read](PVCore::PVTaskProgress& progress) {
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
			    return _type.row_filter_error(
			        *_type.with_paths(*_description, _files->sources().front()), condition);
		    });
		    _filter->show();
		    Q_EMIT ready_changed();
	    });
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
