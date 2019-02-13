/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2018
 */

#ifndef __PVKERNEL_PVWSLHELPER_H__
#define __PVKERNEL_PVWSLHELPER_H__

#include <sys/utsname.h>

#include <QDirIterator>

#include <pvkernel/core/PVUtils.h>

namespace PVCore
{

class PVWSLHelper
{
  private:
	static constexpr const char WSL_WINDOWS_ROOT[] = "/mnt";

  public:
	static bool is_microsoft_wsl()
	{
		struct utsname uname_buf;
		bool is_microsoft_wsl = false;

		uname(&uname_buf);
		is_microsoft_wsl = strstr(uname_buf.release, "Microsoft") != NULL;

		return is_microsoft_wsl;
	}

	static std::string user_directory()
	{
		static const char* userprofile = std::getenv("WSL_USERPROFILE");
		assert(userprofile);
		return userprofile ? userprofile : "";
	}

	static std::vector<std::pair<std::string, std::string>> drives_list()
	{
		std::vector<std::pair<std::string, std::string>> drives_list;

		QDirIterator dir_it(WSL_WINDOWS_ROOT, QDir::Dirs | QDir::NoDotAndDotDot);
		while (dir_it.hasNext()) {
			const QFileInfo& file_info(dir_it.next());
			drives_list.emplace_back(file_info.baseName().toStdString(),
			                         file_info.absoluteFilePath().toStdString());
		}

		return drives_list;
	}
};

} // namespace PVCore

#endif // __PVKERNEL_PVWSLHELPER_H__
