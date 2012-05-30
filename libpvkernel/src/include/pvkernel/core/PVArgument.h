/*! \file PVArgument.h
 * $Id: PVArgument.h 3090 2011-06-09 04:59:46Z stricaud $ Copyright (C)
 * Sébastien Tricaud 2011-2011 Copyright (C) Philippe Saadé 2011-2011 Copyright
 * (C) Picviz Labs 2011
 *
 * \section idea The idea behind PVArgument and friends
 * 
 * The main idea behind PVArgument and PVArgumentList is to define two easy way
 * to defines a list of user-modifiable arguments.  Today, this is mainly used
 * by the filter arguments, the mapping/plotting arguments. Classes that
 * accepts these kind of arguments can be implemented thanks to the \ref
 * PVCore::PVFunctionArgs base class.
 *
 * The goals of this system are:
 * <ul>
 * <li>Easy way to add defines new types and
 *     their associated Qt widgets</li>
 * <li>Provide a widget that can modify a
 *     given PVArgumentList</li>
 * <li>Ability to serialize a PVArgumentList</li>
 * </ul>
 *
 * In this implementation, PVArgument is just a typedef to QVariant. indeed,
 * the QVariant class already provides the first point of our expectations. See
 * below for more information.
 *
 * \section new-type Declaring a new type
 *
 * Many examples already exists (for instance, in
 * libpvkernel/src/include/core/PVAxisIndexType.h), but here is the global
 * process:
 * <ul>
 * <li>Create a class defining your type.</li>
 * <li>Use the Q_DECLARE_METATYPE macro. Be careful, this macro must be used outside of any
 * user-defined namespace (this is because is uses some global structure, see
 * QMetaType's source code for more informations).</li>
 * </ul>
 *
 * Here is an example :
 *
 * \code
 * namespace PVCore {
 *
 * class PVMyNewType { ...  };
 *
 * }
 *
 * }
 *
 * Q_DECLARE_METATYPE(PVCore::PVMyNewType)
 * \endcode
 *
 * \section gui-binding GUI binding
 *
 * Internally, the QItemEditorFactory class is used to bind PVArgument's types
 * to widgets.  See \ref PVInspector::PVArgumentListWidget for more
 * informations.
 *
 * For a quick reference, in order to bind a new type to a filter widget (or
 * mapping/plotting widget), you need to add the corresponding declaration to
 * the \ref PVInspector::PVArgumentListWidget::create_layer_widget_factory
 * and/or \ref
 * PVInspector::PVArgumentListWidget::create_mapping_plotting_factory
 * functions.
 *
 * \section how-to How to use them
 *
 * To create a list of arguments, simply do :
 *
 * \code
 * PVCore::PVArgumentList args;
 * args[PVCore::PVArgumentKey("my_unique_key", "The description of this key")] = QString("I am a QString");
 * args["my_unique_key"] = QString("I am another QString");
 * \endcode
 *
 * Note that you need to provide the description of the key only once, and that
 * an argument inside a PVArgumentList can be indexed thanks to a
 * PVCore::PVArgumentKey, or directly by a string representing the key. Thus,
 * each key will only have one description available, so be careful to choose unique key names.
 * See \ref PVCore::PVArgumentKey for more informations on how this works and current
 * limitations.
 *
 * In order to create a dialog box that would ask the user to modify these
 * arguments (considering this arguments being used by a filter, associated to
 * an existing Picviz::PVView), you can use of the helper functions of
 * PVInspector::PVArgumentListWidget :
 *
 * \code
 * Picviz::PVView* view;
 * [..]
 * // Get a factory object for our Picviz::PVView object
 * QItemEditorFactory* item_factory = PVInspector::PVArgumentListWidget::create_layer_widget_factory(*view);
 * if (PVInspector::PVArgumentListWidget::modify_arguments_dlg(item_factory, args, this)) {
 *    // Argument have been modified, go on...
 * }
 * \endcode
 *
 * Please note that this is not the only way to do it, please refer to \ref PVInspector::PVArgumentListWidget for
 * more informations.
 *
 */

#ifndef PVCORE_PVARGUMENT_H
#define PVCORE_PVARGUMENT_H

#include <pvkernel/core/general.h>
#include <QHash>
#include <QString>
#include <QDomElement>
#include <QVariant>

/*!
 */

namespace PVCore {

/*! \brief PVArgument key that can be used as a QHash key.
 *
 * See \ref PVArgument.h documentation for a complete description of the argument system.
 *
 * \todo The association between a key and its description uses a non thread-safe QHash. For now, this is not an issue,
 * but could become in a close futur.
 * \todo We should be able to create std::map<PVArgumentKey, PVArgument> objects, or any other containers that uses
 * comparaison operations. Thus, it just means "implement operator<" :)
 */
class LibKernelDecl PVArgumentKey: public QString
{
public:
	PVArgumentKey(QString const& key, QString const& desc = QString()):
		QString(key),
		_desc(desc)
	{
		if (desc.isNull()) {
			set_desc_from_key();
		}
		else {
			_key_desc[key] = desc;
		}
	}
	PVArgumentKey(const char* key):
		QString(key)
	{
		set_desc_from_key();
	}

	inline QString const& key() const { return *((QString*)this); }
	inline QString const& desc() const { return _desc; }

private:
	void set_desc_from_key()
	{
		_desc = _key_desc.value(*this, *this);
	}
private:
	QString _desc;
	static QHash<QString, QString> _key_desc;
};


}

extern unsigned int LibKernelDecl qHash(PVCore::PVArgumentKey const& key);

namespace PVCore {

typedef QVariant                           PVArgument;
typedef QHash<PVArgumentKey,PVArgument>    PVArgumentList;
typedef QList<PVArgumentList::key_type>    PVArgumentKeyList;

//class PVArgumentList : public QHash<PVArgumentKey, PVArgument>
//{
//public:
//	int remove(const PVArgumentKey& key);
//	iterator insert(const PVArgumentKey& key, const PVArgument & value);
//
//private:
//	QList<PVArgumentKey> _ordered_keys;
//};

class PVArgumentTypeBase
{
	public:
		PVArgumentTypeBase() {};
		virtual ~PVArgumentTypeBase() {};
	public:
		virtual bool is_equal(const PVArgumentTypeBase &other) const = 0;
		virtual QString to_string() const = 0;
		virtual PVArgument from_string(QString const& str, bool* ok = 0) const = 0;
		virtual void serialize(QDataStream& out) const
		{
			out << to_string();
		}
		virtual PVArgument unserialize(QDataStream& in) const
		{
			QString str;
			in >> str;
			return from_string(str);
		}
};

template <class T>
class PVArgumentType: public PVArgumentTypeBase
{
	virtual bool is_equal(const PVArgumentTypeBase &other) const
	{
		const T* pother = dynamic_cast<const T*>(&other);
		if (!pother) {
			return false;
		}
		return *((T*)this) == *pother;
	}
};


QDataStream &operator<<(QDataStream &out, const PVArgumentTypeBase &obj);
QDataStream &operator>>(QDataStream &in, const PVArgumentTypeBase &obj);

LibKernelDecl QString PVArgument_to_QString(PVArgument const& v);
LibKernelDecl PVArgument QString_to_PVArgument(const QString &s, const QVariant& v, bool* res_ok = 0);

LibKernelDecl void PVArgumentList_to_QSettings(const PVArgumentList& args, QSettings& settings, const QString& group_name);
LibKernelDecl PVArgumentList QSettings_to_PVArgumentList(QSettings& settings, const PVArgumentList& def_args, const QString& group_name);

LibKernelDecl void PVArgumentList_to_QDomElement(const PVArgumentList& args, QDomElement& elt);
LibKernelDecl PVArgumentList QDomElement_to_PVArgumentList(QDomElement const& elt, const PVArgumentList& def_args);

LibKernelDecl void dump_argument_list(PVArgumentList const& l);

LibKernelDecl PVCore::PVArgumentList filter_argument_list_with_keys(PVArgumentList const& args, PVArgumentKeyList const& keys, PVArgumentList const& def_args);

void PVArgumentList_set_common_args_from(PVCore::PVArgumentList& ret, PVCore::PVArgumentList const& ref);

}


#endif
