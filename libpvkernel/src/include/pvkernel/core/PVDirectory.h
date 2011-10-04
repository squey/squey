#ifndef PVCORE_PVDIRECTORY_H
#define PVCORE_PVDIRECTORY_H

#include <pvkernel/core/general.h>
#include <QString>

namespace PVCore {

class LibKernelDecl PVDirectory
{
public:
	static bool remove_rec(QString const& dirName);
};

}

#endif
