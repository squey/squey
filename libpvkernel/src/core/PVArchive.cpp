/**
 * \file PVArchive.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <archive.h>
#include <archive_entry.h>

#include <pvkernel/core/PVArchive.h>

#include <QDir>
#include <QDirIterator>

static void picviz_archive_read_support(struct archive* a)
{
	// Support all formats
	archive_read_support_format_all(a);
	//archive_read_support_format_raw(a);

	// Support all compression schemes but UUE
	// (cf. issue #163 and http://groups.google.com/group/libarchive-discuss/browse_thread/thread/641feadda4ff94b1)
	archive_read_support_compression_bzip2(a);
	archive_read_support_compression_compress(a);
	archive_read_support_compression_gzip(a);
	archive_read_support_compression_lzma(a);
	archive_read_support_compression_xz(a);
	archive_read_support_compression_rpm(a);
	archive_clear_error(a);
}

static void picviz_archive_read_support_noformat(struct archive* a)
{
	// Support all formats
	archive_read_support_format_raw(a);

	archive_read_support_compression_bzip2(a);
	archive_read_support_compression_gzip(a);
	archive_clear_error(a);
}

static int copy_data(struct archive *ar, struct archive *aw)
{
	int r;
	const void *buff;
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
	bool ret;
	struct archive* a;
	struct archive_entry *entry;

	QByteArray path_local = path.toLocal8Bit();
	const char* filename = path_local.constData();

	a = archive_read_new();
	picviz_archive_read_support(a);
	ret = archive_read_open_file(a, filename, 1000) == ARCHIVE_OK;
	if (!ret) {
		archive_read_close(a);
		return false;
	}
	ret = archive_read_next_header(a, &entry) == ARCHIVE_OK;
	archive_read_close(a);

	if (!ret) {
		a = archive_read_new();
		picviz_archive_read_support_noformat(a);
		ret = archive_read_open_file(a, filename, 1000) == ARCHIVE_OK;
		if (ret) {
			ret = archive_read_next_header(a, &entry) == ARCHIVE_OK;
		}
		archive_read_close(a);
		int ac = archive_compression(a);
		if (ret && ac == 0) {
			ret = false;
		}
	}

	return ret;
}

bool PVCore::PVArchive::extract(QString const& path, QString const& dir_dest, QStringList &extracted_files)
{
	struct archive* a;
	struct archive* ext;
	struct archive_entry *entry;
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
			PVLOG_WARN("Unable to create the directory %s for extraction !\n", qPrintable(dir_dest));
			return false;
		}
	}



	// Convert the path to local encoding, and use it in libarchive
	QByteArray path_local = path.toLocal8Bit();
	const char* filename = path_local.constData();

	flags = ARCHIVE_EXTRACT_TIME;
	flags |= ARCHIVE_EXTRACT_SECURE_NODOTDOT;

	a = archive_read_new();
	picviz_archive_read_support(a);
	ext = archive_write_disk_new();
	archive_write_disk_set_options(ext, flags);
	archive_write_disk_set_standard_lookup(ext);
	if ((r = archive_read_open_file(a, filename, 10240))) {
		return false;
	}
	r = archive_read_next_header(a, &entry);
	bool read_raw = false;
	if (r != ARCHIVE_OK) {
		archive_read_close(a);
		a = archive_read_new();
		picviz_archive_read_support_noformat(a);
		archive_read_open_file(a, filename, 10240);
		r = archive_read_next_header(a, &entry);
		read_raw = true;
	}

	for (;;) {
		if (r == ARCHIVE_EOF)
			break;
		if (r != ARCHIVE_OK) {
			PVLOG_ERROR("Error while extracting archive %s: %s\n", filename, archive_error_string(a));
			return false;
		}
		if (r < ARCHIVE_WARN) {
			return false;
		}
		QString qentry(archive_entry_pathname(entry));
		qentry = qentry.trimmed();
		if (qentry.startsWith(QChar('/')) || qentry.startsWith(QChar('\\'))) {
			qentry = qentry.mid(1,-1);
		}
		path_extract = qdir_dest.cleanPath(qdir_dest.absoluteFilePath(qentry));
		path_extract_local = path_extract.toLocal8Bit();
		filename_ext = path_extract_local.constData();

		//PVLOG_INFO("Extract %s from %s to %s...\n", archive_entry_pathname(entry), filename, filename_ext);
		archive_entry_set_pathname(entry, filename_ext);
		if (read_raw) {
			archive_entry_set_perm(entry, 0400);
		}
		r = archive_write_header(ext, entry);
		if (r != ARCHIVE_OK) {
			PVLOG_ERROR("Error while extracting archive %s: %s\n", filename, archive_error_string(a));
		}
		else
		if (archive_entry_size(entry) > 0 || read_raw) {
			r = copy_data(a, ext);
			if (r != ARCHIVE_OK) {
				PVLOG_ERROR("Error while extracting archive %s: %s\n", filename, archive_error_string(a));
				return false;
			}
			if (r < ARCHIVE_WARN) {
				return false;
			}
		}
		r = archive_write_finish_entry(ext);
		if (r != ARCHIVE_OK) {
			PVLOG_ERROR("Error while extracting archive %s: %s\n", filename, archive_error_string(a));
			return false;
		}
		if (r < ARCHIVE_WARN) {
			return false;
		}
		extracted_files.push_back(path_extract);

		r = archive_read_next_header(a, &entry);
	}
	archive_read_finish(a);
	archive_write_finish(ext);
	
	return true;

}

bool PVCore::PVArchive::create_tarbz2(QString const& ar_path, QString const& dir_path)
{
	struct archive *a;
	struct archive_entry *entry;
	struct stat st;
	char buff[8192];
	int len;

	QDir dir(dir_path);
	QString dir_path_abs = dir.canonicalPath();

	a = archive_write_new();
	archive_write_set_format_ustar(a);
	if (archive_write_set_compression_gzip(a) != ARCHIVE_OK) {
		PVLOG_ERROR("Unable to use GZIP compression\n");
		return false;
	}
	QByteArray ar_path_ba = ar_path.toLocal8Bit();
	if (archive_write_open_filename(a, ar_path_ba.constData()) != ARCHIVE_OK) {
		PVLOG_ERROR("Unable to open file %s: %s.\n", ar_path_ba.constData(), archive_error_string(a));
		return false;
	}

	QDirIterator it(dir_path_abs, QDir::Files | QDir::NoSymLinks | QDir::NoDotAndDotDot | QDir::Hidden, QDirIterator::Subdirectories);
	while (it.hasNext()) {
		it.next();
		QString path = it.fileInfo().canonicalFilePath();
		stat(qPrintable(path), &st);
		QString ar_en_path = path.mid(dir_path_abs.size());
		while (ar_en_path.at(0) == PICVIZ_PATH_SEPARATOR_CHAR) {
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
		len = f.read(buff, sizeof(buff));
		while ( len > 0 ) {
			archive_write_data(a, buff, len);
			len = f.read(buff, sizeof(buff));
		}
		f.close();
		archive_entry_free(entry);
	}
	archive_write_close(a);
	archive_write_finish(a);

	return true;
}
