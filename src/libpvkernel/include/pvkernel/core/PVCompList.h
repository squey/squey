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

#ifndef PVCOMPLIST_H
#define PVCOMPLIST_H

#include <pvkernel/core/PVArgument.h>
template <class Key, class T>
class QHash;

namespace PVCore
{

template <class K, class V>
bool comp_hash(QHash<K, V> const& h1, QHash<K, V> const& h2)
{
	if (h1.count() != h2.count()) {
		return false;
	}

	for (auto it1 = h1.constBegin(); it1 != h1.constEnd(); it1++) {
		auto it2 = h2.find(it1.key());
		if (it1.value() != it2.value()) {
			return false;
		}
	}

	return true;
}

bool comp_hash(PVCore::PVArgumentList const& h1, PVCore::PVArgumentList const& h2);

bool comp_hash(PVCore::PVArgumentList const& h1,
               PVCore::PVArgumentList const& h2,
               const PVCore::PVArgumentKeyList& keys);
} // namespace PVCore

#endif
