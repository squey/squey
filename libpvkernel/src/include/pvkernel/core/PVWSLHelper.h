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
		is_microsoft_wsl = strcasestr(uname_buf.release, "Microsoft") != NULL;

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
