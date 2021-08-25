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

#include <pvkernel/core/PVLogger.h>
#include <pvkernel/core/qobject_helpers.h>
#include <pvkernel/core/PVClassLibrary.h>

#include <pvdisplays/PVDisplaysImpl.h>
#include <pvdisplays/PVDisplaysContainer.h>

PVDisplays::PVDisplaysImpl* PVDisplays::PVDisplaysImpl::_instance = nullptr;

static const char* plugins_get_displays_dir()
{
	const char* pluginsdir;

	// FIXME : This is dead code
	pluginsdir = getenv("INENDI_DISPLAYS_DIR");

	return pluginsdir;
}

PVDisplays::PVDisplaysImpl& PVDisplays::PVDisplaysImpl::get()
{
	static PVDisplaysImpl instance;
	return instance;
}

void PVDisplays::PVDisplaysImpl::load_plugins()
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(
	    QString(plugins_get_displays_dir()), "libdisplay");
	if (ret == 0) {
		PVLOG_WARN("No display plugins have been loaded !\n");
	} else {
		PVLOG_INFO("%d display plugins have been loaded.\n", ret);
	}
}
