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

#ifndef LIBPVPCAP_H
#define LIBPVPCAP_H

#include <rapidjson/allocators.h>
#include <rapidjson/rapidjson.h>
#include <vector>
#include <string>
#include <QDir>
#include <QStandardPaths>

#include "rapidjson/document.h"

/*!
 * Namespace that groups together the components of managing
 * the tree of protocols in a pcap file.
 */
namespace pvpcap
{

// use DEL character as separator as it is very unlikely to be present and is XML 1.0 compliant
static constexpr const char SEPARATOR[] = "\x7F";
static constexpr const char QUOTE[] = "n";


std::string tshark_path();

//! The start of what tshark wrote on its standard error to a file, without its warning to root.
std::string tshark_errors(const std::string& path);

} // namespace pvpcap

#endif // LIBPVPCAP_H
