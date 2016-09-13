/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVARCHIVE_H
#define PVCORE_PVARCHIVE_H

#include <stdexcept>

class QStringList;
class QString;

namespace PVCore
{

class ArchiveCreationFail : public std::runtime_error
{
  public:
	using std::runtime_error::runtime_error;
};

class ArchiveUncompressFail : public std::runtime_error
{
  public:
	using std::runtime_error::runtime_error;
};

/*! \brief Provide conveniant functions around libarchive (http://www.libarchive.com)
 *
 * This class provides static functions for conveniant uses of the libarchive interfaces.
 *
 * \todo We might consider outsourcing some of these as this might be useful for the community (see
 *libarchive's mailing-list archive).
 */
namespace PVArchive
{
/* \brief Detects if a file can be processed as an archive thanks to libarchive.
 * \param[in] path Path to the potential archive file
 *
 * This function uses libarchive mecanics to detect if a file is an archive.
 */
bool is_archive(QString const& path);

/* \brief Extract an archive supported by libarchive.
 * \param[in]  path            Path to the archive file
 * \param[in]  dir_dest        Destination directory where the archive will be extracted
 * \param[out] extracted_files Relative path (to dir_dest) of the files extracted
 * \return true if the extraction succeded, false otherwise.
 *
 * This function extracts the archive pointed by path into dir_dest.
 */
void extract(QString const& path, QString const& dir_dest, QStringList& extracted_files);

/* \brief Create a tar/bz2 archive of a directory
 * \param[in] ar_path  Path to the archive to create
 * \param[in] dir_path Path of the directory to include into the archive.
 * \return true if the archive was successfully created, false otherwise.
 *
 * \note This is mainly used by PVCore::PVSerializeArchiveZip
 */
void create_tarbz2(QString const& ar_path, QString const& dir_path);
};
}

#endif
