/**
 * \file PVArchive.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PVARCHIVE_H
#define PVCORE_PVARCHIVE_H

#include <pvkernel/core/general.h>

#include <QStringList>
#include <QString>

namespace PVCore {

/*! \brief Provide conveniant functions around libarchive (http://www.libarchive.com)
 *
 * This class provides static functions for conveniant uses of the libarchive interfaces.
 *
 * \todo We might consider outsourcing some of these as this might be useful for the community (see libarchive's mailing-list archive).
 */
class LibKernelDecl PVArchive
{
public:
	/* \brief Detects if a file can be processed as an archive thanks to libarchive.
	 * \param[in] path Path to the potential archive file
	 *
	 * This function uses libarchive mecanics to detect if a file is an archive.
	 */
	static bool is_archive(QString const& path);

	/* \brief Extract an archive supported by libarchive.
	 * \param[in]  path            Path to the archive file
	 * \param[in]  dir_dest        Destination directory where the archive will be extracted
	 * \param[out] extracted_files Relative path (to dir_dest) of the files extracted
	 * \return true if the extraction succeded, false otherwise.
	 *
	 * This function extracts the archive pointed by path into dir_dest.
	 */
	static bool extract(QString const& path, QString const& dir_dest, QStringList &extracted_files);

	/* \brief Create a tar/bz2 archive of a directory
	 * \param[in] ar_path  Path to the archive to create
	 * \param[in] dir_path Path of the directory to include into the archive. 
	 * \return true if the archive was successfully created, false otherwise.
	 *
	 * \note This is mainly used by PVCore::PVSerializeArchiveZip
	 */
	static bool create_tarbz2(QString const& ar_path, QString const& dir_path);
};

}

#endif
