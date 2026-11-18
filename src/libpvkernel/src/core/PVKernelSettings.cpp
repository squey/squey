//
// MIT License
//
// © Squey, 2026
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

#include <pvkernel/core/PVKernelSettings.h>

#include <pvkernel/rush/PVNraw.h>

#include <QByteArray>

#include <climits>

namespace
{

std::vector<PVCore::PVSettingChoice> color_schemes()
{
	std::vector<PVCore::PVSettingChoice> schemes{{"dark", "Dark"}, {"light", "Light"}};
	// macOS has no way to tell the scheme of the system yet.
#ifndef __APPLE__
	if (qgetenv("DISABLE_FOLLOW_SYSTEM_THEME").isEmpty()) {
		schemes.push_back({"system", "Follow the system"});
	}
#endif
	return schemes;
}

} // namespace

const PVCore::PVSetting<QString> PVCore::settings::color_scheme(
    {.key = "appearance.color_scheme",
     .type = PVSettingType::Choice,
     .default_value = "dark",
     .title = "Color scheme",
     .description = "Colors of the interface. Following the system switches them along with "
                    "the desktop.",
     .choices = color_schemes(),
     .keywords = {"theme", "dark", "light", "theme_scheme"}});

const PVCore::PVSetting<QString> PVCore::settings::working_directory(
    {.key = "general.working_directory",
     .type = PVSettingType::Path,
     .default_value = PVRush::PVNraw::default_tmp_path,
     .title = "Working directory",
     .description = "Where Squey keeps the data it imports, and the temporary files of the SQL "
                    "console and of the investigations it saves. A fast disk makes Squey "
                    "faster. Squey creates a directory named after the user in it.",
     .requires_restart = true,
     .keywords = {"temporary", "cache", "disk", "nraw_tmp"}});

const PVCore::PVSetting<int> PVCore::settings::recent_items(
    {.key = "general.recent_items",
     .type = PVSettingType::Int,
     .default_value = 30,
     .title = "Recent items",
     .description = "Sources and formats each list of the start screen keeps. The list of the "
                    "investigations keeps them all.",
     .minimum = 1,
     .maximum = 1000,
     .keywords = {"history", "start screen"}});

const PVCore::PVSetting<int> PVCore::settings::import_chunk_size(
    {.key = "import.chunk_size",
     .type = PVSettingType::Int,
     .default_value = 10048576,
     .title = "Chunk size",
     .description = "Bytes of a text or pcap file that the import reads at once. Larger chunks "
                    "take more memory.",
     .minimum = 4096,
     .maximum = INT_MAX,
     .unit = "bytes",
     .keywords = {"memory", "max_size_chunk"}});

const PVCore::PVSetting<int> PVCore::settings::import_parallel_chunks(
    {.key = "import.parallel_chunks",
     .type = PVSettingType::Int,
     .default_value = 0,
     .title = "Chunks in progress",
     .description = "Chunks the import works on at once. More chunks take more memory without "
                    "importing faster. 0 works on as many chunks as the processor runs threads.",
     .minimum = 0,
     .maximum = 4096,
     .keywords = {"threads", "memory", "number_living_chunks"}});

const PVCore::PVSetting<bool> PVCore::settings::force_cpu(
    {.key = "rendering.force_cpu",
     .type = PVSettingType::Bool,
     .default_value = false,
     .title = "Draw with the processor",
     .description = "Draw the lines of the parallel coordinates with the processor rather than "
                    "the graphics card, whose driver may draw them wrongly or crash. Drawing is "
                    "slower.",
     .requires_restart = true,
     .keywords = {"OpenCL", "GPU", "CPU", "graphics card"}});

const PVCore::PVSetting<int> PVCore::settings::opencl_platform(
    {.key = "rendering.opencl_platform",
     .type = PVSettingType::Int,
     .default_value = -1,
     .title = "OpenCL platform",
     .description = "Which of the OpenCL platforms able to draw the lines of the parallel "
                    "coordinates draws them, counted from 0 in the order of the OpenCL loader. "
                    "-1 picks one, dedicated graphics cards first.",
     .minimum = -1,
     .maximum = 255,
     .requires_restart = true,
     .keywords = {"GPU", "graphics card", "platform_index"}});
