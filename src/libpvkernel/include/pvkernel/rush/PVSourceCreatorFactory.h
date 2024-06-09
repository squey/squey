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

#ifndef PVRUSH_PVSOURCECREATORFACTORY_H
#define PVRUSH_PVSOURCECREATORFACTORY_H

#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/core/PVArgument.h>
#include <list>
#include <map>
#include <QHash>
#include <QString>
#include <utility>

#include "pvkernel/rush/PVFormat.h"
#include "pvkernel/rush/PVInputDescription.h"

namespace PVRush
{

typedef std::list<PVSourceCreator_p> list_creators;
typedef std::pair<PVFormat, PVSourceCreator_p> pair_format_creator;
typedef QHash<QString, pair_format_creator> hash_format_creator;

class PVSourceCreatorFactory
{
  public:
	static PVSourceCreator_p get_by_input_type(PVInputType_p in_t);
	static float discover_input(pair_format_creator format,
	                            PVInputDescription_p input,
	                            bool* cancellation = nullptr);
};
} // namespace PVRush

#endif
