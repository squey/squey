//
// MIT License
//
// © ESI Group, 2015
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

#include <pvkernel/widgets/PVExportDlg.h>

#include <pvkernel/core/PVStreamingCompressor.h>
#include <pvkernel/widgets/PVCSVExporterWidget.h>
#include <pvkernel/widgets/PVFileDialog.h>

#include <QComboBox>
#include <QDialogButtonBox>
#include <QDir>
#include <QFormLayout>
#include <QGroupBox>
#include <QPushButton>
#include <QStackedWidget>
#include <QStandardPaths>
#include <QVBoxLayout>

#include <cassert>
#include <regex>
#include <string>

PVWidgets::PVExportDlg::PVExportDlg(QWidget* parent /* = nullptr */,
                                    Target target /* = Target::File */,
                                    PVExporterWidgetInterface* csv_widget /* = nullptr */)
    : QDialog(parent), _target(target), _formats(new QComboBox), _pages(new QStackedWidget)
{
	setWindowTitle(tr("Export selection"));
	_formats->setObjectName("export_format");
	connect(_formats, &QComboBox::currentTextChanged, this, [this](QString const& filter) {
		_pages->setCurrentWidget(_exporters.value(filter).second);
	});

	if (csv_widget == nullptr) {
		csv_widget = new PVCSVExporterWidget();
	}
	PVRush::PVExporterBase& csv_exporter = csv_widget->exporter();
	insert_format(_formats->count(), ".csv files (*.csv)", csv_widget, csv_exporter);
	for (const std::string& extension : PVCore::PVStreamingCompressor::supported_extensions()) {
		const QString& filter =
		    QString::fromStdString(".csv." + extension + " files (*.csv." + extension + ")");
		insert_format(_formats->count(), filter, csv_widget, csv_exporter);
	}

	// The gzip format by default.
	for (int index = _formats->count() - 1; index >= 0; index--) {
		if (_formats->itemText(index).contains("gz")) {
			_formats->setCurrentIndex(index);
			break;
		}
	}

	auto* format_layout = new QFormLayout;
	format_layout->addRow(tr("Format:"), _formats);

	auto* options = new QGroupBox;
	auto* options_layout = new QVBoxLayout(options);
	options_layout->addWidget(_pages);

	auto* buttons = new QDialogButtonBox(QDialogButtonBox::Cancel);
	QPushButton* export_button = buttons->addButton(tr("Export..."), QDialogButtonBox::AcceptRole);
	export_button->setObjectName("export_button");
	export_button->setDefault(true);
	connect(buttons, &QDialogButtonBox::accepted, this, &PVExportDlg::choose_path);
	connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

	auto* layout = new QVBoxLayout(this);
	layout->addLayout(format_layout);
	layout->addWidget(options);
	layout->addWidget(buttons);

	// Rather than Cancel, the first button of the box, while Enter exports.
	_formats->setFocus();
}

QStringList PVWidgets::PVExportDlg::name_filters() const
{
	QStringList filters;
	for (int index = 0; index < _formats->count(); index++) {
		filters << _formats->itemText(index);
	}
	return filters;
}

QString PVWidgets::PVExportDlg::selected_name_filter() const
{
	return _formats->currentText();
}

void PVWidgets::PVExportDlg::select_name_filter(QString const& filter)
{
	if (const int index = _formats->findText(filter); index >= 0) {
		_formats->setCurrentIndex(index);
	}
}

QString PVWidgets::PVExportDlg::file_extension() const
{
	const std::string& str = selected_name_filter().toStdString();
	const std::regex base_regex(R"(.*\(\*(.*)\))");
	std::smatch base_match;
	std::regex_match(str, base_match, base_regex);
	assert(base_match.length() == 1);
	return QString::fromStdString(base_match[1].str());
}

PVRush::PVExporterBase& PVWidgets::PVExportDlg::exporter()
{
	return *_exporters.value(selected_name_filter()).first;
}

void PVWidgets::PVExportDlg::add_format(QString const& filter,
                                        QWidget* page,
                                        PVRush::PVExporterBase& exporter)
{
	insert_format(_formats_ahead++, filter, page, exporter);
}

void PVWidgets::PVExportDlg::insert_format(int index,
                                           QString const& filter,
                                           QWidget* page,
                                           PVRush::PVExporterBase& exporter)
{
	_exporters.insert(filter, {&exporter, page});
	if (_pages->indexOf(page) < 0) {
		_pages->addWidget(page);
	}
	_formats->insertItem(index, filter);
}

void PVWidgets::PVExportDlg::choose_path()
{
	const QString home = QStandardPaths::writableLocation(QStandardPaths::HomeLocation);
	// The previous choice when the file it named could not be written. A suggested name keeps
	// the extension where nothing adds it, as in the Flatpak.
	const QString path =
	    _target == Target::Directory
	        ? PVFileDialog::getExistingDirectory(this, windowTitle(),
	                                             _path.isEmpty() ? home : _path)
	        : PVFileDialog::getSaveFileName(
	              this, windowTitle(),
	              _path.isEmpty() ? QDir(home).filePath(_file_name + file_extension()) : _path,
	              selected_name_filter(), nullptr, {}, file_extension());
	if (not path.isEmpty()) {
		_path = path;
		accept();
	}
}
