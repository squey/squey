/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVInputTypeFilename.h"

#include <pvbase/general.h>

#include <pvkernel/core/PVArchive.h>
#include <pvkernel/core/PVDirectory.h>

#include <QMessageBox>
#include <QFileInfo>

#include <sys/time.h>
#include <sys/resource.h>

PVRush::PVInputTypeFilename::PVInputTypeFilename() : PVInputTypeDesc<PVFileDescription>()
{
	struct rlimit rlim;
	if (getrlimit(RLIMIT_NOFILE, &rlim) != 0) {
		PVLOG_WARN("Unable to get nofile limit. Uses 1024 by default.\n");
		_limit_nfds = 1024;
	} else {
		_limit_nfds =
		    rlim.rlim_cur - 1; // Take the soft limit as this is the one that will limit us...
	}
}

bool PVRush::PVInputTypeFilename::createWidget(hash_formats const& formats,
                                               hash_formats& /*new_formats*/,
                                               list_inputs& inputs,
                                               QString& format,
                                               PVCore::PVArgumentList& /*args_ext*/,
                                               QWidget* parent) const
{
	QStringList formats_name = formats.keys();
	formats_name.sort();

	formats_name.prepend(QString(INENDI_AUTOMATIC_FORMAT_STR));
	formats_name.prepend(QString(INENDI_BROWSE_FORMAT_STR));
	formats_name.prepend(QString(INENDI_LOCAL_FORMAT_STR));

	// Get information from file dialog
	PVImportFileDialog file_dlg(formats_name, parent);
	QStringList filenames = file_dlg.getFileNames(format);

	return load_files(filenames, inputs, parent);
}

bool PVRush::PVInputTypeFilename::load_files(QStringList const& filenames,
                                             list_inputs& inputs,
                                             QWidget* parent) const
{
	for (QString const& filename : filenames) {
		inputs.push_back(PVInputDescription_p(new PVFileDescription(filename)));
	}

	if (inputs.size() >= _limit_nfds - 200) {
		ssize_t nopen = _limit_nfds - 200;
		if (nopen <= 0) {
			nopen = 1;
		}
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

	return inputs.size() > 0;
}

PVRush::PVInputTypeFilename::~PVInputTypeFilename()
{
	for (int i = 0; i < _tmp_dir_to_delete.size(); i++) {
		PVLOG_INFO("Delete temporary directory %s...\n", qPrintable(_tmp_dir_to_delete[i]));
		PVCore::PVDirectory::remove_rec(_tmp_dir_to_delete[i]);
	}
}

QString PVRush::PVInputTypeFilename::name() const
{
	return QString("file");
}

QString PVRush::PVInputTypeFilename::human_name() const
{
	return QString("File import plugin");
}

QString PVRush::PVInputTypeFilename::human_name_serialize() const
{
	return QString(tr("Local files"));
}

QString PVRush::PVInputTypeFilename::internal_name() const
{
	return QString("00-file");
}

QString PVRush::PVInputTypeFilename::menu_input_name() const
{
	return QString("Local files...");
}

QString PVRush::PVInputTypeFilename::tab_name_of_inputs(list_inputs const& in) const
{
	QString tab_name;
	PVFileDescription* f = dynamic_cast<PVFileDescription*>(in[0].get());
	assert(f);
	QFileInfo fi(f->path());
	if (in.count() == 1) {
		tab_name = fi.fileName();
	} else {
		tab_name = fi.canonicalPath();
	}
	return tab_name;
}

bool PVRush::PVInputTypeFilename::get_custom_formats(PVInputDescription_p in,
                                                     hash_formats& formats) const
{
	// Two types of custom format: inendi.format/picviz.format exists in the directory of the file,
	// or file + ".format" exists
	bool res = false;
	PVFileDescription* f = dynamic_cast<PVFileDescription*>(in.get());
	assert(f);
	QString path_custom_format = f->path() + QString(".format");
	QFileInfo fi(path_custom_format);
	QString format_custom_name = fi.fileName();

	if (fi.exists() && fi.isReadable()) {
		formats[format_custom_name] = PVRush::PVFormat(format_custom_name, path_custom_format);
		return true;
	}

	static std::vector<QString> custom_filenames = {"inendi.format", "picviz.format"};

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

QKeySequence PVRush::PVInputTypeFilename::menu_shortcut() const
{
	return QKeySequence::Italic;
}
