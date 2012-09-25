/**
 * \file PVInputType.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PICVIZ_PVINPUTTYPE_H
#define PICVIZ_PVINPUTTYPE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/core/PVRegistrableClass.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <QList>
#include <QKeySequence>
#include <QObject>
#include <QDomDocument>
#include <QIcon>
#include <QCursor>

namespace PVRush {

class LibKernelDecl PVInputType: public QObject, public PVCore::PVRegistrableClass<PVInputType>
{
	Q_OBJECT
public:
	typedef boost::shared_ptr<PVInputType> p_type;
	// List of inputs description
	typedef QList<PVInputDescription_p> list_inputs;
	typedef list_inputs list_inputs_desc;
public:
	virtual bool createWidget(hash_formats const& formats, hash_formats& new_formats, list_inputs &inputs, QString& format, PVCore::PVArgumentList& args_ext, QWidget* parent = NULL) const = 0;
	virtual QString name() const = 0;
	virtual QString human_name() const = 0;
	virtual QString human_name_serialize() const = 0;
	// Warning: the "human name" of an input must be *unique* accross all the possible inputs
	virtual QString human_name_of_input(PVInputDescription_p in) const { return in->human_name(); };
	virtual QString menu_input_name() const = 0;
	virtual QKeySequence menu_shortcut() const { return QKeySequence(); }
	virtual QString tab_name_of_inputs(list_inputs const& in) const = 0;
	virtual bool get_custom_formats(PVInputDescription_p in, hash_formats &formats) const = 0;
	virtual PVCore::PVSerializeObject_p serialize_inputs(PVCore::PVSerializeObject& obj, QString const& name, list_inputs& inputs) const = 0;
	virtual void serialize_inputs_ref(PVCore::PVSerializeObject& obj, QString const& name, list_inputs& inputs, PVCore::PVSerializeObject_p so_ref) const = 0;

	virtual QIcon icon() const { return QIcon(); }
	virtual QCursor cursor() const { return QCursor(); }

	virtual void save_input_to_qsettings(const PVInputDescription& input_descr, QSettings& settings) = 0;
	virtual PVInputDescription_p load_input_from_qsettings(const QSettings& settings) = 0;

public:
	QStringList human_name_of_inputs(list_inputs const& in) const
	{
		QStringList ret;
		list_inputs::const_iterator it;
		for (it = in.begin(); it != in.end(); it++) {
			ret << human_name_of_input(*it);
		}
		return ret;
	}
public:
	void edit_format(QString const& path, QWidget* parent) const
	{
		emit edit_format_signal(path, parent);
	}

	void edit_format(QDomDocument& doc, QWidget* parent) const
	{
		emit edit_format_signal(doc, parent);
	}

	void connect_parent(QObject const* parent) const
	{
		connect((QObject*) this, SIGNAL(edit_format_signal(QString const&, QWidget*)), parent, SLOT(edit_format_Slot(QString const&, QWidget*)));
		connect((QObject*) this, SIGNAL(edit_format_signal(QDomDocument&, QWidget*)), parent, SLOT(edit_format_Slot(QDomDocument&, QWidget*)));
	}
signals:
	void edit_format_signal(QString const& path, QWidget* parent) const;
	void edit_format_signal(QDomDocument& doc, QWidget* parent) const;
};

template <typename T>
class PVInputTypeDesc: public PVInputType
{
public:
	virtual PVCore::PVSerializeObject_p serialize_inputs(PVCore::PVSerializeObject& obj, QString const& name, list_inputs& inputs) const
	{
		// Get name of inputs
		QStringList descs;
		list_inputs::const_iterator it;
		for (it = inputs.begin(); it != inputs.end(); it++) {
			descs << human_name_of_input(*it);
		}
		return obj.list<list_inputs, boost::shared_ptr<T> >(name, inputs, human_name_serialize(), NULL, descs);
	}
	virtual void serialize_inputs_ref(PVCore::PVSerializeObject& obj, QString const& name, list_inputs& inputs, PVCore::PVSerializeObject_p so_ref) const
	{
		obj.list_ref(name, inputs, so_ref);
	}

	virtual void save_input_to_qsettings(const PVInputDescription& input_descr, QSettings& settings)
	{
		input_descr.save_to_qsettings(settings);
	}

	virtual PVInputDescription_p load_input_from_qsettings(const QSettings& settings)
	{
		PVInputDescription_p input_descr_p(new T());
		input_descr_p->load_from_qsettings(settings);

		return input_descr_p;
	}
};

typedef PVInputType::p_type PVInputType_p;

}

//#define REGISTER_INPUT_TYPE(T) REGISTER_CLASS(T().name(), T())
#ifdef WIN32
LibKernelDeclExplicitTempl PVCore::PVClassLibrary<PVRush::PVInputType>;
#endif

#endif
