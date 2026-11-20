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

#include "PVTextImportWidget.h"
#include "PVInputTypeFilename.h"

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVTaskProgress.h>
#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/rush/PVFormatDetection.h>
#include <pvkernel/rush/PVSourceCreatorFactory.h>
#include <pvkernel/widgets/PVFileDialog.h>
#include <pvkernel/widgets/PVImportFilesWidget.h>

#include <QButtonGroup>
#include <QFile>
#include <QFileInfo>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QRadioButton>
#include <QVBoxLayout>

#include <memory>
#include <stdexcept>

PVRush::PVTextImportWidget::PVTextImportWidget(PVInputTypeFilename const& type, QWidget* parent)
    : PVImportWidget(parent)
    , _type(type)
    , _files(new PVWidgets::PVImportFilesWidget(tr("Import text files"), tr("All files (*)")))
    , _detected(new QRadioButton(tr("The one next to the files, else detected from their lines")))
    , _from_file(new QRadioButton(tr("From a file:")))
    , _format_path(new QLineEdit)
    , _made(new QRadioButton(tr("Made in the format builder")))
    , _edit(new QPushButton(tr("Edit in the format builder...")))
{
	_detected->setObjectName("import_format_detected");
	_from_file->setObjectName("import_format_from_file");
	_format_path->setObjectName("import_format_path");
	_made->setObjectName("import_format_made");
	_edit->setObjectName("import_format_edit");
	_made->hide();
	_edit->setEnabled(false);
	_detected->setChecked(true);
	auto* browse = new QPushButton(tr("Browse..."));

	// A group of their own: left to their auto-exclusivity, the radio buttons of the files, in the
	// same widget, kept checking one of these from unchecking the others.
	auto* formats = new QButtonGroup(this);
	formats->addButton(_detected);
	formats->addButton(_from_file);
	formats->addButton(_made);
	auto* format_layout = new QGridLayout;
	format_layout->addWidget(_detected, 0, 0, 1, 3);
	format_layout->addWidget(_from_file, 1, 0);
	format_layout->addWidget(_format_path, 1, 1);
	format_layout->addWidget(browse, 1, 2);
	format_layout->addWidget(_made, 2, 0, 1, 3);
	auto* edit_layout = new QHBoxLayout;
	edit_layout->addWidget(_edit);
	edit_layout->addStretch();

	auto* layout = new QVBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);
	layout->addWidget(_files);
	layout->addWidget(new QLabel(tr("Format")));
	layout->addLayout(format_layout);
	layout->addLayout(edit_layout);

	connect(_files, &PVWidgets::PVImportFilesWidget::paths_changed, this, [this]() {
		// Formats and investigations, which a dialog of all files lets the user choose along.
		QStringList kept;
		for (const QString& path : _files->paths()) {
			const QString suffix = QFileInfo(path).suffix().toLower();
			if (suffix != "format" and suffix != "pvi") {
				kept << path;
			}
		}
		if (kept != _files->paths()) {
			_files->set_paths(kept);
		}
		_edit->setEnabled(not kept.isEmpty());
		Q_EMIT ready_changed();
	});
	connect(_from_file, &QRadioButton::toggled, this, &PVImportWidget::ready_changed);
	connect(_made, &QRadioButton::toggled, this, &PVImportWidget::ready_changed);
	connect(_format_path, &QLineEdit::textEdited, this, [this]() {
		_from_file->setChecked(true);
		Q_EMIT ready_changed();
	});
	connect(browse, &QPushButton::clicked, this, &PVTextImportWidget::choose_format);
	connect(_edit, &QPushButton::clicked, this, &PVTextImportWidget::edit_format);
}

bool PVRush::PVTextImportWidget::is_ready() const
{
	return not _files->paths().isEmpty() and
	       (not _from_file->isChecked() or not _format_path->text().isEmpty());
}

std::vector<PVWidgets::PVImportWidget::source_t> PVRush::PVTextImportWidget::sources() const
{
	std::vector<source_t> sources;
	for (const QStringList& paths : _files->sources()) {
		if (paths.size() > _type.max_files_per_source()) {
			throw std::runtime_error(
			    tr("A source of %1 files opens more files than the system lets Squey open at once: "
			       "import them as one source each, or raise the limit of open files.")
			        .arg(paths.size())
			        .toStdString());
		}
		QList<PVInputDescription_p> inputs;
		for (const QString& path : paths) {
			inputs << PVInputDescription_p(new PVFileDescription(path, paths.size() > 1));
		}
		if (_made->isChecked()) {
			sources.emplace_back(inputs, PVFormat(_made_format.documentElement()));
		} else {
			sources.emplace_back(inputs, _from_file->isChecked() ? format_of(_format_path->text())
			                                                     : format_next_to(inputs));
		}
	}
	return sources;
}

void PVRush::PVTextImportWidget::set_paths(QStringList const& paths)
{
	_files->set_paths(paths);
}

void PVRush::PVTextImportWidget::set_format(QString const& path)
{
	_format_path->setText(path);
	_from_file->setChecked(true);
}

void PVRush::PVTextImportWidget::set_edited_format(QDomDocument const& format)
{
	_made_format = format;
	_made->show();
	_made->setChecked(true);
}

void PVRush::PVTextImportWidget::edit_format()
{
	const PVInputDescription_p input(new PVFileDescription(_files->paths().front(), false));
	const PVInputType_p type = LIB_CLASS(PVInputType)::get().get_class_by_name(_type.name());
	QString path;
	if (_made->isChecked()) {
		Q_EMIT format_edit_requested(_made_format, input, type);
		return;
	}
	if (_from_file->isChecked()) {
		path = _format_path->text();
	} else {
		path = format_next_to({input}).get_full_path();
	}
	if (not path.isEmpty()) {
		QFile file(path);
		QDomDocument document;
		if (not file.open(QIODevice::ReadOnly) or not document.setContent(&file)) {
			Q_EMIT message(Severity::critical,
			               tr("The format '%1' cannot be read.").arg(path).toHtmlEscaped());
			return;
		}
		Q_EMIT format_edit_requested(document, input, type);
		return;
	}

	auto detected = std::make_shared<QDomDocument>();
	run_task(
	    tr("Detecting the format of %1").arg(input->human_name()),
	    [detected, input, type](PVCore::PVTaskProgress& progress) {
		    try {
			    *detected = detect_format_document(
			        input, PVSourceCreatorFactory::get_by_input_type(type), &progress);
		    } catch (const PVFormatInvalid&) {
			    // The user makes one from none then.
		    }
	    },
	    [this, detected, input, type](bool completed) {
		    if (completed) {
			    Q_EMIT format_edit_requested(*detected, input, type);
		    }
	    });
}

void PVRush::PVTextImportWidget::choose_format()
{
	const QString path = PVWidgets::PVFileDialog::getOpenFileName(
	    this, tr("Load format from..."),
	    _format_path->text().isEmpty() ? QString() : QFileInfo(_format_path->text()).absolutePath(),
	    tr("Formats (*.format)"));
	if (not path.isEmpty()) {
		set_format(path);
	}
}

PVRush::PVFormat
PVRush::PVTextImportWidget::format_next_to(QList<PVInputDescription_p> const& inputs) const
{
	hash_formats found;
	for (const PVInputDescription_p& input : inputs) {
		_type.get_custom_formats(input, found);
	}
	// Several, the files do not agree on one.
	return found.size() == 1 ? found.first() : PVFormat();
}
