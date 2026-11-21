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

#include "PVInputTypeFilename.h"
#include "PVTextImportWidget.h"

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVFormatDetection.h>
#include <pvkernel/rush/PVSourceCreatorFactory.h>

#include <pvbase/general.h>

#include <pvkernel/core/PVArchive.h>
#include <pvkernel/core/PVDirectory.h>

#include <pvkernel/widgets/PVFileDialog.h>

#include <QMessageBox>
#include <QFileInfo>
#include <QStandardPaths>

#include <ranges>

#include <sys/time.h>
//#include <sys/resource.h>

PVRush::PVInputTypeFilename::PVInputTypeFilename() : PVInputTypeDesc<PVFileDescription>()
{
	// struct rlimit rlim;
	// if (getrlimit(RLIMIT_NOFILE, &rlim) != 0) {
	// 	PVLOG_WARN("Unable to get nofile limit. Uses 1024 by default.\n");
		_limit_nfds = 1024;
	// } else {
	// 	_limit_nfds =
	// 	    rlim.rlim_cur - 1; // Take the soft limit as this is the one that will limit us...
	// }
}

bool PVRush::PVInputTypeFilename::create_widget(hash_formats& /*formats*/,
                                                list_inputs& inputs,
                                                QString& /*format*/,
                                                PVCore::PVArgumentList& /*args_ext*/,
                                                QWidget* parent) const
{
	const QStringList filenames =
	    PVWidgets::PVFileDialog::getOpenFileNames(
	        parent, QObject::tr("Import text files"),
	        QStandardPaths::writableLocation(QStandardPaths::HomeLocation)) |
	    std::views::filter([](const QString& f) {
		    QString ext = QFileInfo(f).suffix().toLower();
		    return ext != "format" and ext != "pvi";
	    }) |
	    std::ranges::to<QStringList>();

	return load_files(filenames, inputs, parent);
}

PVWidgets::PVImportWidget* PVRush::PVInputTypeFilename::create_import_widget(QWidget* parent) const
{
	return new PVTextImportWidget(*this, parent);
}

PVRush::PVFormat
PVRush::PVInputTypeFilename::get_format_from_inputs(list_inputs const& inputs,
                                                    PVCore::PVTaskProgress* progress) const
{
	// The format of the first file alone: that of a source of several files names them by itself.
	const auto& first = dynamic_cast<PVFileDescription const&>(*inputs.front());
	return detect_format(PVInputDescription_p(new PVFileDescription(first.path(), false)),
	                     PVSourceCreatorFactory::get_by_input_type(
	                         LIB_CLASS(PVInputType)::get().get_class_by_name(name())),
	                     progress);
}

bool PVRush::PVInputTypeFilename::load_files(
    QStringList const& filenames,
    list_inputs& inputs,
    QWidget* parent) const
{
	for (QString const& filename : filenames) {
		inputs.push_back(
		    PVInputDescription_p(new PVFileDescription(filename, filenames.size() > 1)));
	}

#ifdef __linux__
	if ((uint64_t)inputs.size() >= (uint64_t)_limit_nfds - 200) {
		uint64_t nopen = (uint64_t)_limit_nfds - 200;
		QString msg =
		    QObject::tr("You are trying to open %1 files, and your system limits a user to open %2 "
		                "file descriptor at once.\nConsidering the needs of the application, this "
		                "value must be set to a higher value. In order to change this limit, edit "
		                "/etc/security/limits.conf and add the following lines:")
		        .arg(inputs.size())
		        .arg(_limit_nfds);
		msg += "\n\n*\tsoft\tnofile\t131070\n*\thard\tnofile\t131070\n\n";
		msg += QObject::tr("You can set 131070 to a bigger value if needed. Then, you need to "
		                   "logout and login for these changes to be effectives.");
		msg += "\n\n";
		msg += QObject::tr("Only the first %1 file(s) will be opened.").arg(nopen);
		QMessageBox err(QMessageBox::Warning, QObject::tr("Too many files selected"), msg,
		                QMessageBox::Ok, parent);
		err.exec();
		inputs.erase(inputs.begin() + nopen + 1, inputs.end());
	}
#else
	(void) parent;
#endif

	return inputs.size() > 0;
}

PVRush::PVInputTypeFilename::~PVInputTypeFilename()
{
	for (auto & i : _tmp_dir_to_delete) {
		PVLOG_INFO("Delete temporary directory %s...\n", qPrintable(i));
		PVCore::PVDirectory::remove_rec(i);
	}
}

QString PVRush::PVInputTypeFilename::name() const
{
	return {"file"};
}

QString PVRush::PVInputTypeFilename::human_name() const
{
	return {"File import plugin"};
}

QString PVRush::PVInputTypeFilename::human_name_serialize() const
{
	return QString(tr("Local files"));
}

QString PVRush::PVInputTypeFilename::internal_name() const
{
	return {"00-file"};
}

QStringList PVRush::PVInputTypeFilename::get_supported_extensions() const
{
    return { "csv", "csv.zip", "csv.gz", "csv.bz2", "csv.xz", "csv.zst",
             "tsv", "tsv.zip", "tsv.gz", "tsv.bz2", "tsv.xz", "tsv.zst" };
}

QString PVRush::PVInputTypeFilename::menu_input_name() const
{
	return {"Text"};
}

QString PVRush::PVInputTypeFilename::tab_name_of_inputs(list_inputs const& in) const
{
	QString tab_name;
	auto* f = dynamic_cast<PVFileDescription*>(in[0].get());
	assert(f);
	QFileInfo fi(f->path());
	if (in.count() == 1) {
		tab_name = fi.fileName();
	} else {
		tab_name = fi.canonicalPath();
	}
	return tab_name;
}

bool PVRush::PVInputTypeFilename::get_custom_formats(
    PVInputDescription_p in,
    hash_formats& formats) const
{
	// Three types of custom format: squey.format/inendi.format/picviz.format exists in the directory of the file,
	// or file + ".format" exists
	bool res = false;
	auto* f = dynamic_cast<PVFileDescription*>(in.get());
	assert(f);
	QString path_custom_format = f->path() + QString(".format");
	QFileInfo fi(path_custom_format);
	QString format_custom_name = fi.fileName();

	if (fi.exists() && fi.isReadable()) {
		formats[format_custom_name] = PVRush::PVFormat(format_custom_name, path_custom_format);
		return true;
	}

	static std::vector<QString> custom_filenames = {"squey.format", "inendi.format", "picviz.format"};

	QDir d = fi.dir();

	auto path_custom_dir_format = std::find_if(
	    custom_filenames.begin(), custom_filenames.end(), [&d](const QString& filename) {
		    QFileInfo fi = QFileInfo(d.absoluteFilePath(filename));
		    return fi.exists() && fi.isReadable();
	    });

	if (path_custom_dir_format == custom_filenames.end()) {
		return res;
	}

	format_custom_name = "custom_directory:" + d.path();
	if (!formats.contains(format_custom_name)) {
		formats[format_custom_name] =
		    PVRush::PVFormat(format_custom_name, d.absoluteFilePath(*path_custom_dir_format));
	}

	return true;
}
