#include <pvkernel/core/PVArchive.h>
#include <pvkernel/core/PVDirectory.h>
#include <pvkernel/core/PVSerializeArchiveZip.h>

PVCore::PVSerializeArchiveZip::PVSerializeArchiveZip(version_t v):
	PVSerializeArchive(v)
{
}

PVCore::PVSerializeArchiveZip::PVSerializeArchiveZip(QString const& zip_path, archive_mode mode, version_t v):
	PVSerializeArchive(v)
{
	open_zip(zip_path, mode);
}

PVCore::PVSerializeArchiveZip::~PVSerializeArchiveZip()
{
	if (_is_opened) {
		finish();
	}
}

void PVCore::PVSerializeArchiveZip::open_zip(QString const& zip_path, archive_mode mode)
{
	// Generate a temporary directory
	_tmp_path = PVCore::PVDirectory::temp_dir(zip_path + "-XXXXXX");
	if (_tmp_path.isEmpty()) {
		throw PVSerializeArchiveError("Unable to create a temporary directory.");
	}

	if (mode == read) {
		// If we are reading an archive, then we need to extract it
		// to a temporary place, and use this for PVSerializeArchive !
		QStringList files;
		if (!PVArchive::extract(zip_path, _tmp_path, files)) {
			throw PVSerializeArchiveError(QString("Unable to extract archive '%1'.").arg(zip_path));
		}
	}

	PVSerializeArchive::open(_tmp_path, mode);
	_zip_path = zip_path;
}

void PVCore::PVSerializeArchiveZip::finish()
{
	PVSerializeArchive::finish();
	if (_zip_path.isEmpty()) {
		return;
	}

	if (is_writing()) {
		// Create the archive from _tmp_path
		PVArchive::create_tarbz2(_zip_path, _tmp_path);
	}

	// Delete the temporary folder
	PVDirectory::remove_rec(_tmp_path);
}
