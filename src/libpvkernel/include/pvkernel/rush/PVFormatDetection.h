/* * MIT License
 *
 * © Squey, 2026
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

#ifndef __PVRUSH_PVFORMATDETECTION_H__
#define __PVRUSH_PVFORMATDETECTION_H__

#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/rush/PVSourceCreator.h>

#include <pvbase/types.h>

#include <QDomDocument>
#include <QDomElement>

#include <cstddef>

namespace PVCore
{
class PVTaskProgress;
} // namespace PVCore

namespace PVRush
{

class PVTypesDiscoveryOutput;

/**
 * The format of the splitter @p splitter, which cuts the lines after the first @p preamble_lines
 * into @p axes_count fields, each an axis of strings, as the detection of a format starts from.
 */
QDomDocument format_of_splitter(PVFilter::PVFieldsSplitter const& splitter,
                                PVCol axes_count,
                                size_t preamble_lines);

/** How many rows the discovery of the types of @p axes_count axes reads. */
PVRow types_discovery_row_count(PVCol axes_count);

/**
 * Gives the axes of the format @p format_root the types that @p discovered found for them, and
 * the names it found in a header if @p name_axes.
 *
 * @return whether it found a header
 */
bool set_discovered_types(QDomElement format_root,
                          PVTypesDiscoveryOutput const& discovered,
                          bool name_axes);

/**
 * The format that Squey finds for the text @p input, which @p creator reads: the splitter that
 * cuts its lines into fields, the types of the fields from the first rows, and their names from
 * a header, which the format skips. Out of the GUI thread, it may take long, while @p progress,
 * if any, shows how far it got and can stop it: the format is then empty.
 *
 * @throw PVFormatInvalid if no splitter cuts the lines of the input
 */
PVFormat detect_format(PVInputDescription_p const& input,
                       PVSourceCreator_p const& creator,
                       PVCore::PVTaskProgress* progress = nullptr);

} // namespace PVRush

#endif // __PVRUSH_PVFORMATDETECTION_H__
