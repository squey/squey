/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QSpacerItem>
#include <QStackedLayout>

#include <pvkernel/widgets/PVExportDlg.h>
#include <pvkernel/widgets/PVCSVExporterWidget.h>

#include <pvkernel/core/PVProgressBox.h>

#include <regex>

PVWidgets::PVExportDlg::PVExportDlg(
	QWidget* parent /* = 0 */,
	QFileDialog::AcceptMode accept_mode /* = QFileDialog::AcceptSave */,
	QFileDialog::FileMode file_mode /* = QFileDialog::AnyFile*/
)
	: PVWidgets::PVFileDialog(parent)
{
	// Set this flags to make sure we can access the layout.
	setOption(QFileDialog::DontUseNativeDialog);
	setWindowTitle(tr("Export selection"));
	setAcceptMode(accept_mode);
	setFileMode(file_mode);

	// Use default CSV exporter if input type doesn't define a specific one
	_exporter_widget = new PVWidgets::PVCSVExporterWidget();

	// Get the layout from the QFileDialog
	// Layout is:
	// -----------------main_layout--------------------------------
	// |           name bar where you can put filename/filedir    |
	// ------------------------------------------------------------
	// | dir tree | list of files                                 |
	// ------------------------------------------------------------
	// |filename_label | filename | valide_button                 |
	// ------------------------------------------------------------
	// |filetype_label | filetype | cancel_button                 |
	// ------------------------------------------------------------
	// |----------------group_box---------------------------------|
	// ||                  export_layout                         ||
	// |----------------------------------------------------------|
	// ------------------------------------------------------------
	QGridLayout* main_layout = static_cast<QGridLayout*>(layout());

	_groupbox = new QGroupBox();
	main_layout->addWidget(_groupbox, 4, 0, 1, 3);

	// Add input specific exporter filter string if any
	// Default CSV exporter
	_name_filters << ".csv files (*.csv)";
	for (const std::string& extension : PVCore::PVStreamingCompressor::supported_extensions()) {
		const QString& name_filter =
		    QString::fromStdString(".csv." + extension + " files (*.csv." + extension + ")");
		_name_filters << name_filter;
	}
	setNameFilters(_name_filters);
	auto suffix_from_filter = [](const QString& filter) { return filter.split(" ")[0]; };

	// Set specific exporter as default if any, .csv.gz otherwise
	int gz_idx;
	for (gz_idx = _name_filters.size() - 1; gz_idx >= 0; gz_idx--) {
		if (_name_filters[gz_idx].contains("gz")) {
			break;
		}
	}
	int default_filter_index = gz_idx;

	QStackedLayout* stacked_layout = new QStackedLayout;
	stacked_layout->addWidget(_exporter_widget);
	_groupbox->setLayout(stacked_layout);

	_filter_selected_f = [=,this](const QString& filter) {
		setDefaultSuffix(suffix_from_filter(filter));

		// force filters to reset as setting 'QFileDialog::Directory' erase them
		setNameFilters(_name_filters);
		selectNameFilter(filter);		
	};
	_conn = QObject::connect(this, &QFileDialog::filterSelected, _filter_selected_f);
	_filter_selected_f(_name_filters.at(default_filter_index));
}

QString PVWidgets::PVExportDlg::file_extension() const
{
	const std::string& str = selectedNameFilter().toStdString();
	const std::regex base_regex(".*\\(\\*(.*)\\)");
	std::smatch base_match;
	std::regex_match(str, base_match, base_regex);
	assert(base_match.length() == 1);
	return QString::fromStdString(base_match[1].str());
}