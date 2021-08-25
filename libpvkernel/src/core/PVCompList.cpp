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

#include <pvkernel/core/PVCompList.h>
#include <pvkernel/core/PVArgument.h>   // for PVArgumentList, PVArgument, etc
#include <pvkernel/core/PVOrderedMap.h> // for PVOrderedMapNode

#include <vector> // for vector

#include <QMetaType>
#include <QVariant>

namespace PVCore
{

bool comp_hash(PVCore::PVArgumentList const& h1, PVCore::PVArgumentList const& h2)
{
	if (h1.size() != h2.size()) {
		return false;
	}

	for (const auto& it1 : h1) {
		auto it2 = h2.find(it1.key());
		if (it2 == h2.end()) {
			return false;
		}
		if (it1.value().userType() >= QMetaType::User &&
		    it1.value().userType() == it2->value().userType()) { // custom type
			const PVArgumentTypeBase* v1 =
			    static_cast<const PVArgumentTypeBase*>(it1.value().constData());
			const PVArgumentTypeBase* v2 =
			    static_cast<const PVArgumentTypeBase*>(it2->value().constData());
			if (!v1->is_equal(*v2)) {
				return false;
			}
		} else if (it1.value() != it2->value()) {
			return false;
		}
	}

	return true;
}

bool comp_hash(PVCore::PVArgumentList const& h1,
               PVCore::PVArgumentList const& h2,
               const PVCore::PVArgumentKeyList& keys)
{
	for (PVCore::PVArgumentKey key : keys) {
		PVCore::PVArgument arg1 = h1.at(key);
		PVCore::PVArgument arg2 = h2.at(key);

		if (!arg1.isValid() || !arg2.isValid()) {
			return false;
		}

		if (arg1.userType() >= QMetaType::User &&
		    arg1.userType() == arg2.userType()) { // custom type
			const PVArgumentTypeBase* v1 = static_cast<const PVArgumentTypeBase*>(arg1.constData());
			const PVArgumentTypeBase* v2 = static_cast<const PVArgumentTypeBase*>(arg2.constData());
			if (!v1->is_equal(*v2)) {
				return false;
			}
		} else if (arg1 != arg2) {
			return false;
		}
	}

	return true;
}
} // namespace PVCore
