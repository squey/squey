/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
} // namespace PVArchive
;
} // namespace PVCore

#endif
