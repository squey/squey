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

#include <pvkernel/core/PVArchive.h>   // for PVArchive
#include <pvkernel/core/PVDirectory.h> // for remove_rec, temp_dir
#include <pvkernel/core/PVLogger.h>
#include <pvkernel/core/PVSerializeArchive.h> // for PVSerializeArchive, etc
#include <pvkernel/core/PVSerializeArchiveExceptions.h>
#include <pvkernel/core/PVSerializeArchiveZip.h>
#include <qcontainerfwd.h>
#include <qhash.h>
#include <qlist.h>
#include <qsettings.h>
#include <string> // for allocator, operator+, etc
#include <QString>

#include <pvlogger.h> // FIXME: REMOVEME

PVCore::PVSerializeArchiveZip::PVSerializeArchiveZip(version_t v, bool save_log_file)
    : PVSerializeArchive(v, save_log_file)
{
}

PVCore::PVSerializeArchiveZip::PVSerializeArchiveZip(QString const& zip_path,
                                                     archive_mode mode,
                                                     version_t v,
                                                     bool save_log_file)
    : PVSerializeArchive(v, save_log_file)
{
	open_zip(zip_path, mode);
}

PVCore::PVSerializeArchiveZip::~PVSerializeArchiveZip()
{
	try {
		close_zip();
	} catch (PVCore::ArchiveCreationFail const& e) {
		std::string error_msg = std::string("Zip file not saved:") + e.what();
		PVLOG_ERROR(error_msg.c_str());
	}
}

void PVCore::PVSerializeArchiveZip::close_zip()
{
	if (_is_opened) {
		for (auto it = _objs_attributes.constBegin(); it != _objs_attributes.constEnd(); it++) {
			it.value()->sync();
		}
		if (_zip_path.isEmpty()) {
			return;
		}

		if (is_writing()) {
			// Create the archive from _tmp_path
			PVArchive::create_tarbz2(_zip_path, _tmp_path);
		}

		// Delete the temporary folder
		PVDirectory::remove_rec(_tmp_path);
		_is_opened = false;
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
		try {
			PVArchive::extract(zip_path, _tmp_path, files);
		} catch (PVCore::ArchiveUncompressFail const& e) {
			throw PVSerializeArchiveError("Unable to extract archive '" + zip_path.toStdString() +
			                              "'.");
		}
	}

	PVSerializeArchive::open(_tmp_path, mode);
	_zip_path = zip_path;
}
