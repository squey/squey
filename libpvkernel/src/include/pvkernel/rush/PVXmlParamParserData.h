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

#ifndef PVXMLPARAMPARSERDATA_H
#define PVXMLPARAMPARSERDATA_H
#include <QString>

#include <pvkernel/core/PVArgument.h>
#include <pvkernel/core/PVCompList.h>
#include <pvkernel/filter/PVFieldsFilter.h>

namespace PVRush
{

class PVXmlParamParserData
{

  public:
	PVXmlParamParserData()
	{
		// By default, the number of children is unknown.
		nchildren = 0;
	}
	int axis_id;
	PVFilter::PVFieldsFilterReg_p filter_lib;
	PVCore::PVArgumentList filter_args;
	size_t nchildren;
};
} // namespace PVRush
#endif /* PVXMLPARAMPARSERDATA_H */
