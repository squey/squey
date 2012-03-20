#include <pvkernel/core/PVCompList.h>
#include <pvkernel/core/PVTimeFormatType.h>

namespace PVCore {

template <>
bool comp_list(QList<PVArgument> const& l1, QList<PVArgument> const& l2)
{
	if (l1.size() != l2.size()) {
		return false;
	}

	QList<QVariant>::const_iterator it1,it2;
	it1 = l1.begin();
	it2 = l2.begin();

	for (; it1 != l1.end(); it1++) {
		QVariant const& v1 = *it1;
		QVariant const& v2 = *it2;

		/////
//		if (v1.canConvert<PVTimeFormatType>() &&
//		    v2.canConvert<PVTimeFormatType>()) {
//			if (v1.value<PVTimeFormatType>() != v2.value<PVTimeFormatType>()) {
//				return false;
//			}
		/////

		if (v1.userType() >= QMetaType::User && v1.userType() == v2.userType()) { // custom type
			const PVArgumentTypeBase* v1b = static_cast<const PVArgumentTypeBase*>(v1.constData());
			const PVArgumentTypeBase* v2b = static_cast<const PVArgumentTypeBase*>(v2.constData());
			if (!v1b->is_equal(*v2b)) {
				return false;
			}
		}
		else
		if (v1 != v2) {
			return false;
		}
		it2++;
	}

	return true;
}

}
