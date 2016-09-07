/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVArchive.h>   // for PVArchive
#include <pvkernel/core/PVDirectory.h> // for PVDirectory
#include <pvkernel/core/PVLogger.h>    // for PVLOG_ERROR, PVLOG_INFO, etc

#include <pvbase/general.h> // for INENDI_PATH_SEPARATOR_CHAR

#include <archive.h>       // for archive_error_string, etc
#include <archive_entry.h> // for archive_entry_set_pathname, etc

#include <boost/thread/condition_variable.hpp>

#include <array>      // for array
#include <cstddef>    // for size_t
#include <sys/stat.h> // for stat, off_t, st_atime, etc

#include <QChar>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QString>
#include <QStringList>

namespace boost
{
class thread_interrupted;
}

static void inendi_archive_read_support(struct archive* a)
{
	// Support all formats
	archive_read_support_format_all(a);
	// archive_read_support_format_raw(a);

	// Support all compression schemes but UUE
	// (cf. issue #163 and
	// http://groups.google.com/group/libarchive-discuss/browse_thread/thread/641feadda4ff94b1)
	archive_read_support_filter_bzip2(a);
	archive_read_support_filter_compress(a);
	archive_read_support_filter_gzip(a);
	archive_read_support_filter_lzma(a);
	archive_read_support_filter_xz(a);
	archive_read_support_filter_rpm(a);
	archive_clear_error(a);
}

static void inendi_archive_read_support_noformat(struct archive* a)
{
	// Support all formats
	archive_read_support_format_raw(a);

	archive_read_support_filter_bzip2(a);
	archive_read_support_filter_gzip(a);
	archive_clear_error(a);
}

static int copy_data(struct archive* ar, struct archive* aw)
{
	int r;
	const void* buff;
	size_t size;
	off_t offset;

	for (;;) {
		r = archive_read_data_block(ar, &buff, &size, &offset);
		if (r == ARCHIVE_EOF) {
			return (ARCHIVE_OK);
		}
		if (r != ARCHIVE_OK) {
			return (r);
		}
		r = archive_write_data_block(aw, buff, size, offset);
		if (r != ARCHIVE_OK) {
			PVLOG_ERROR("Error while extracting archive: %s\n", archive_error_string(ar));
			return (r);
		}
	}
}

bool PVCore::PVArchive::is_archive(QString const& path)
{
	static QStringList supported_archives{"gz", "bz2", "zip", "lzma", "xz", "z"};

	// use only the file suffix (the "last" extension), not the complete one
	return supported_archives.contains(QFileInfo(path).suffix());
}

bool PVCore::PVArchive::extract(QString const& path,
                                QString const& dir_dest,
                                QStringList& extracted_files)
{
	struct archive* a;
	struct archive* ext;
	struct archive_entry* entry;
	int flags;
	int r;
	const char* filename_ext;
	QString path_extract;
	QByteArray path_extract_local;

	// Parse the destination directory
	QDir qdir_dest(dir_dest);
	if (!qdir_dest.exists()) {
		// Try to create it
		if (!QDir().mkpath(dir_dest)) {
			PVLOG_WARN("Unable to create the directory %s for extraction !\n",
			           qPrintable(dir_dest));
			return false;
		}
	}

	// Convert the path to local encoding, and use it in libarchive
	QByteArray path_local = path.toLocal8Bit();
	const char* filename = path_local.constData();

	flags = ARCHIVE_EXTRACT_TIME;
	flags |= ARCHIVE_EXTRACT_SECURE_NODOTDOT;

	a = archive_read_new();
	inendi_archive_read_support(a);
	ext = archive_write_disk_new();
	archive_write_disk_set_options(ext, flags);
	archive_write_disk_set_standard_lookup(ext);
	if ((r = archive_read_open_filename(a, filename, 10240)) != 0) {
		return false;
	}
	r = archive_read_next_header(a, &entry);
	bool read_raw = false;
	if (r != ARCHIVE_OK) {
		archive_read_close(a);
		a = archive_read_new();
		inendi_archive_read_support_noformat(a);
		archive_read_open_filename(a, filename, 10240);
		r = archive_read_next_header(a, &entry);
		read_raw = true;
	}

	try {
		for (;;) {
			if (r == ARCHIVE_EOF) {
				break;
			}
			if (r != ARCHIVE_OK) {
				PVLOG_ERROR("Error while extracting archive %s: %s\n", filename,
				            archive_error_string(a));
				return false;
			}
			if (r < ARCHIVE_WARN) {
				return false;
			}
			QString qentry(archive_entry_pathname(entry));
			qentry = qentry.trimmed();
			if (qentry.startsWith(QChar('/')) || qentry.startsWith(QChar('\\'))) {
				qentry = qentry.mid(1, -1);
			}
			path_extract = qdir_dest.cleanPath(qdir_dest.absoluteFilePath(qentry));
			path_extract_local = path_extract.toLocal8Bit();
			filename_ext = path_extract_local.constData();

			// PVLOG_INFO("Extract %s from %s to %s...\n", archive_entry_pathname(entry), filename,
			// filename_ext);
			archive_entry_set_pathname(entry, filename_ext);
			if (read_raw) {
				archive_entry_set_perm(entry, 0400);
			}
			r = archive_write_header(ext, entry);
			if (r != ARCHIVE_OK) {
				PVLOG_ERROR("Error while extracting archive %s: %s\n", filename,
				            archive_error_string(a));
			} else if (archive_entry_size(entry) > 0 || read_raw) {
				r = copy_data(a, ext);
				if (r != ARCHIVE_OK) {
					PVLOG_ERROR("Error while extracting archive %s: %s\n", filename,
					            archive_error_string(a));
					return false;
				}
				if (r < ARCHIVE_WARN) {
					return false;
				}
			}
			r = archive_write_finish_entry(ext);
			if (r != ARCHIVE_OK) {
				PVLOG_ERROR("Error while extracting archive %s: %s\n", filename,
				            archive_error_string(a));
				return false;
			}
			if (r < ARCHIVE_WARN) {
				return false;
			}
			extracted_files.push_back(path_extract);

			r = archive_read_next_header(a, &entry);
			boost::this_thread::interruption_point();
		}
	} catch (boost::thread_interrupted const& e) {
		PVLOG_INFO("(PVArchive::extract) Extraction canceled.\n");
		// archive_read_finish(a);
		// archive_write_finish(ext);
		PVCore::PVDirectory::remove_rec(dir_dest); // cleanup
		throw e;
	}
	archive_read_free(a);
	archive_write_free(ext);

	return true;
}

bool PVCore::PVArchive::create_tarbz2(QString const& ar_path, QString const& dir_path)
{
	struct archive* a;
	struct archive_entry* entry;
	struct stat st;
	std::array<char, 8192> buff;
	int len;

	QDir dir(dir_path);
	QString dir_path_abs = dir.canonicalPath();

	a = archive_write_new();
	archive_write_set_format_ustar(a);
	if (archive_write_add_filter_gzip(a) != ARCHIVE_OK) {
		PVLOG_ERROR("Unable to use GZIP compression\n");
		return false;
	}
	QByteArray ar_path_ba = ar_path.toLocal8Bit();
	if (archive_write_open_filename(a, ar_path_ba.constData()) != ARCHIVE_OK) {
		PVLOG_ERROR("Unable to open file %s: %s.\n", ar_path_ba.constData(),
		            archive_error_string(a));
		return false;
	}

	QDirIterator it(dir_path_abs,
	                QDir::Files | QDir::NoSymLinks | QDir::NoDotAndDotDot | QDir::Hidden,
	                QDirIterator::Subdirectories);
	try {
		while (it.hasNext()) {
			it.next();
			QString path = it.fileInfo().canonicalFilePath();
			stat(qPrintable(path), &st);
			QString ar_en_path = path.mid(dir_path_abs.size());
			while (ar_en_path.at(0) == INENDI_PATH_SEPARATOR_CHAR) {
				ar_en_path = ar_en_path.mid(1);
			}
			QByteArray ar_en_path_ba = ar_en_path.toLocal8Bit();
			entry = archive_entry_new();
			archive_entry_set_pathname(entry, ar_en_path_ba.constData());
			archive_entry_set_size(entry, st.st_size);
			archive_entry_set_filetype(entry, AE_IFREG);
			archive_entry_set_perm(entry, 0644);
			archive_entry_set_mtime(entry, st.st_mtime, 0);
			archive_entry_set_atime(entry, st.st_atime, 0);
			archive_write_header(a, entry);

			QFile f(path);
			if (!f.open(QIODevice::ReadOnly)) {
				return false;
			}
			len = f.read(buff.data(), buff.size());
			while (len > 0) {
				archive_write_data(a, buff.data(), len);
				len = f.read(buff.data(), buff.size());
			}
			f.close();
			archive_entry_free(entry);
			boost::this_thread::interruption_point();
		}
	} catch (boost::thread_interrupted const& e) {
		PVLOG_INFO("(PVArchive::create_tarbz2) Compression canceled.\n");
		// archive_write_close(a);
		// archive_write_finish(a);
		PVCore::PVDirectory::remove_rec(dir_path); // cleanup
		throw e;
	}
	archive_write_close(a);
	archive_write_free(a);

	return true;
}
