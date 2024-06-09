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

#ifndef PVCORE_PVARGUMENT_H
#define PVCORE_PVARGUMENT_H

#include <pvkernel/core/PVOrderedMap.h>
#include <qdatastream.h>
#include <qhashfunctions.h>
#include <qlist.h>
#include <qmetatype.h>
#include <qstringfwd.h>
#include <vector>
#include <QHash>
#include <QString>
#include <QVariant>

class QDataStream;
class QDomElement;
class QSettings;

/*!
 */

namespace PVCore
{

/*! \brief PVArgument key that can be used as a QHash key.
 *
 * See \ref PVArgument.h documentation for a complete description of the argument system.
 *
 * \todo The association between a key and its description uses a non thread-safe QHash. For now,
 *this is not an issue,
 * but could become in a close futur.
 * \todo We should be able to create std::map<PVArgumentKey, PVArgument> objects, or any other
 *containers that uses
 * comparaison operations. Thus, it just means "implement operator<" :)
 */
class PVArgumentKey : public QString
{
  public:
	PVArgumentKey(QString const& key, QString const& desc = QString()) : QString(key), _desc(desc)
	{
		if (desc.isNull()) {
			_desc = _key_desc.value(*this, *this);
		} else {
			_key_desc[key] = desc;
		}
	}
	PVArgumentKey(const char* key) : PVArgumentKey(QString(key)) {}

	inline QString const& key() const { return *this; }
	inline QString const& desc() const { return _desc; }

  private:
	QString _desc;
	static QHash<QString, QString> _key_desc;
};
} // namespace PVCore

extern unsigned int qHash(PVCore::PVArgumentKey const& key);

namespace PVCore
{

using PVArgument = QVariant;

class PVArgumentList : public PVOrderedMap<PVArgumentKey, PVArgument>
{
  public:
	PVArgumentList() : PVOrderedMap<PVArgumentKey, PVArgument>(), _edit_flag(true) {}

	bool get_edition_flag() const { return _edit_flag; }

	void set_edition_flag(bool e) { _edit_flag = e; }

  private:
	bool _edit_flag;
};

using PVArgumentKeyList = std::vector<PVArgumentList::key_type>;

class PVArgumentTypeBase
{
  public:
	PVArgumentTypeBase() = default;
	;
	virtual ~PVArgumentTypeBase() = default;
	;

  public:
	virtual bool is_equal(const PVArgumentTypeBase& other) const = 0;
	virtual QString to_string() const = 0;
	virtual PVArgument from_string(QString const& str, bool* ok = nullptr) const = 0;
	virtual void serialize(QDataStream& out) const { out << to_string(); }
	virtual PVArgument unserialize(QDataStream& in) const
	{
		QString str;
		in >> str;
		return from_string(str);
	}
};

template <class T>
class PVArgumentType : public PVArgumentTypeBase
{
	bool is_equal(const PVArgumentTypeBase& other) const override
	{
		const T* pother = dynamic_cast<const T*>(&other);
		if (!pother) {
			return false;
		}
		return *((T*)this) == *pother;
	}
};

QDataStream& operator<<(QDataStream& out, const PVArgumentTypeBase& obj);
QDataStream& operator>>(QDataStream& in, const PVArgumentTypeBase& obj);

QString PVArgument_to_QString(PVArgument const& v);
PVArgument QString_to_PVArgument(const QString& s, const QVariant& v, bool* res_ok = nullptr);

void PVArgumentList_to_QSettings(const PVArgumentList& args,
                                 QSettings& settings,
                                 const QString& group_name);
PVArgumentList QSettings_to_PVArgumentList(QSettings& settings,
                                           const PVArgumentList& def_args,
                                           const QString& group_name);

void PVArgumentList_to_QDomElement(const PVArgumentList& args, QDomElement& elt);
PVArgumentList QDomElement_to_PVArgumentList(QDomElement const& elt,
                                             const PVArgumentList& def_args);

void dump_argument_list(PVArgumentList const& l);

PVCore::PVArgumentList filter_argument_list_with_keys(PVArgumentList const& args,
                                                      PVArgumentKeyList const& keys,
                                                      PVArgumentList const& def_args);

void PVArgumentList_set_common_args_from(PVCore::PVArgumentList& ret,
                                         PVCore::PVArgumentList const& ref);
void PVArgumentList_set_missing_args(PVCore::PVArgumentList& ret,
                                     PVCore::PVArgumentList const& def_args);
} // namespace PVCore

#endif
