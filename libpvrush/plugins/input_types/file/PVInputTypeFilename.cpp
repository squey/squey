#include "PVInputTypeFilename.h"
#include "PVImportFileDialog.h"

#include "extract.h"

#include <QMessageBox>
#include <QFileInfo>

#include <stdlib.h>

#ifdef WIN32
#include <io.h>
#include <string.h>
static char* mkdtemp(char* pattern)
{
	errno_t res = _mktemp_s(pattern, strlen(pattern)+1);
	if (res == 0) {
		return pattern;
	}

	return NULL;
}
#endif

// Taken from http://john.nachtimwald.com/2010/06/08/qt-remove-directory-and-its-contents/
static bool removeDir(const QString &dirName)
{
	bool result = true;
	QDir dir(dirName);

	if (dir.exists(dirName)) {
		Q_FOREACH(QFileInfo info, dir.entryInfoList(QDir::NoDotAndDotDot | QDir::System | QDir::Hidden  | QDir::AllDirs | QDir::Files, QDir::DirsFirst)) {
			if (info.isDir()) {
				result = removeDir(info.absoluteFilePath());
			}
			else {
				result = QFile::remove(info.absoluteFilePath());
			}

			if (!result) {
				return result;
			}
		}
		result = dir.rmdir(dirName);
	}

	return result;
}

bool PVRush::PVInputTypeFilename::createWidget(hash_formats const& formats, list_inputs &inputs, QString& format, QWidget* parent) const
{
	QStringList formats_name = formats.keys();
	formats_name.prepend(QString(PICVIZ_AUTOMATIC_FORMAT_STR));
	PVImportFileDialog* dlg = new PVImportFileDialog(formats_name, parent);
	dlg->setDefaults();
	QStringList filenames = dlg->getFileNames(format);
	for (int i = 0; i < filenames.size(); i++) {
		QString const& path = filenames[i];
		bool add_original = true;
		if (is_archive(path)) {
			PVLOG_DEBUG("(import-files) %s is an archive.\n", qPrintable(path));
			QStringList extracted;
			QMessageBox box_ext(QMessageBox::Question,
					            "Import files : archive detected", QString("'%1' has been detected as an archive. Do you want to extract it to a temporary directory and import its content ?").arg(path),
								QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel,
								parent);
			int ret = box_ext.exec();
			switch (ret) {
				case QMessageBox::Yes:
				{
					// Create a temporary directory of name "/tmp/picviz-archivename-XXXXXX"
					QFileInfo fi(path);
					QString tmp_dir_pattern = QDir::temp().absoluteFilePath(QString("picviz-") + fi.fileName() + QString("-XXXXXXXX"));
					QByteArray tmp_dir_ba = tmp_dir_pattern.toLocal8Bit();
					char* tmp_dir_p = mkdtemp(tmp_dir_ba.data());
					if (tmp_dir_p == NULL) {
						PVLOG_WARN("Extraction of %s: unable to create a temporary directory !\n", qPrintable(path));
						break;
					}
					QString tmp_dir = tmp_dir_p;
					_tmp_dir_to_delete.push_back(tmp_dir);
					PVLOG_INFO("Extract archive %s to %s...\n", qPrintable(path), tmp_dir_p);
					if (extract_archive(filenames[i], tmp_dir, extracted)) {
						add_original = false;
						for (int j = 0; j < extracted.count(); j++) {
							inputs.push_back(QVariant(extracted[j]));
						}
					}
					else {
						PVLOG_WARN("Failed to extract archive %s. Loading as a regular file...\n", qPrintable(path));
					}
					break;
				}
				case QMessageBox::Cancel:
					return false;
				case QMessageBox::No:
				default:
					break;
			}
		}

		if (add_original) {
			inputs.push_back(QVariant(path));
		}
	}

	return inputs.size() > 0;
}

PVRush::PVInputTypeFilename::~PVInputTypeFilename()
{
	for (int i = 0; i < _tmp_dir_to_delete.size(); i++) {
		PVLOG_INFO("Delete temporary directory %s...\n", qPrintable(_tmp_dir_to_delete[i]));
		removeDir(_tmp_dir_to_delete[i]);
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

QString PVRush::PVInputTypeFilename::human_name_of_input(PVFilter::PVArgument const& in) const
{
	return in.toString();
}

QString PVRush::PVInputTypeFilename::menu_input_name() const
{
	return QString("Import files...");
}

QString PVRush::PVInputTypeFilename::tab_name_of_inputs(list_inputs const& in) const
{
	QString tab_name;
	QFileInfo fi(in[0].toString());
	if (in.count() == 1) {
		tab_name = fi.fileName();
	}
	else {
		tab_name = fi.canonicalPath();
	}
	return tab_name;
}

bool PVRush::PVInputTypeFilename::get_custom_formats(PVFilter::PVArgument const& in, hash_formats &formats) const
{
	// Two types of custom format: picviz.format exist in the directory of the file,
	// or file + ".format" exists
	bool res = false;
	QString path_custom_format = in.toString() + QString(".format");
	QFileInfo fi(path_custom_format);
	QString format_custom_name = "custom:" + fi.fileName();

	if (QFile(path_custom_format).exists()) {
		formats[format_custom_name] = PVRush::PVFormat(format_custom_name, path_custom_format);
		res = true;
	}

	QDir d = fi.dir();
	QString path_custom_dir_format = d.absoluteFilePath("picviz.format");
	if (!QFile(path_custom_dir_format).exists())
		return res;

	format_custom_name = "custom_directory:" + d.path();
	if (!formats.contains(format_custom_name)) {
		formats[format_custom_name] = PVRush::PVFormat(format_custom_name, path_custom_dir_format);
	}

	return true;
}

QKeySequence PVRush::PVInputTypeFilename::menu_shortcut() const
{
	return QKeySequence::Italic;
}
