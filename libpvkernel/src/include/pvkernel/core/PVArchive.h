#ifndef PVCORE_PVARCHIVE_H
#define PVCORE_PVARCHIVE_H

#include <pvkernel/core/general.h>

#include <QStringList>
#include <QString>

namespace PVCore {

class LibKernelDecl PVArchive
{
public:
	static bool is_archive(QString const& path);
	static bool extract(QString const& path, QString const& dir_dest, QStringList &extracted_files);
	static bool create_tarbz2(QString const& ar_path, QString const& dir_path);
};

}

#endif
