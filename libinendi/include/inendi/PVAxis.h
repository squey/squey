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

#ifndef INENDI_PVAXIS_H
#define INENDI_PVAXIS_H

#include <pvkernel/rush/PVAxisFormat.h> // for PVAxisFormat, etc

#include <pvkernel/core/PVArgument.h> // for PVArgumentList

#include <stdexcept> // for runtime_error

namespace PVCore
{
class PVSerializeObject;
} // namespace PVCore

namespace Inendi
{

/**
 * Exception raised when mapping/plotting combination is invalid.
 */
struct InvalidPlottingMapping : public std::runtime_error {
	using std::runtime_error::runtime_error;
};

/**
 * \class PVAxis
 */
class PVAxis : public PVRush::PVAxisFormat
{
	friend class PVCore::PVSerializeObject;

  public:
	/**
	 * Constructor
	 */
	PVAxis()
	    : PVRush::PVAxisFormat(
	          PVCol(-1)){}; // We have to keep this Ugly constructor as we use QVector
	                        // which perform a lot of
	                        // default construction
	explicit PVAxis(PVRush::PVAxisFormat axis_format);

  public:
	PVCore::PVArgumentList const& get_args_mapping() const { return _args_mapping; }
	PVCore::PVArgumentList const& get_args_plotting() const { return _args_plotting; }

  private:
	static PVCore::PVArgumentList args_from_node(node_args_t const& args_str,
	                                             PVCore::PVArgumentList const& def_args);

  private:
	PVCore::PVArgumentList _args_mapping;
	PVCore::PVArgumentList _args_plotting;
};
} // namespace Inendi

#endif /* INENDI_PVAXIS_H */
