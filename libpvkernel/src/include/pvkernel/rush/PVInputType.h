/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVINPUTTYPE_H
#define INENDI_PVINPUTTYPE_H

#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/core/PVRegistrableClass.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVNraw.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/widgets/PVExporterWidgetInterface.h>
#include <QList>
#include <QKeySequence>
#include <QObject>
#include <QDomDocument>
#include <QIcon>
#include <QCursor>
#include <QSettings>
#include "PVExporter.h"

namespace PVRush
{

class PVInputType : public QObject, public PVCore::PVRegistrableClass<PVInputType>
{
	Q_OBJECT
  public:
	typedef std::shared_ptr<PVInputType> p_type;
	// List of inputs description
	typedef QList<PVInputDescription_p> list_inputs;
	typedef list_inputs list_inputs_desc;

  public:
	virtual bool createWidget(hash_formats const& formats,
	                          hash_formats& new_formats,
	                          list_inputs& inputs,
	                          QString& format,
	                          PVCore::PVArgumentList& args_ext,
	                          QWidget* parent = nullptr) const = 0;

	/* exporter */
	virtual std::unique_ptr<PVRush::PVExporterBase>
	create_exporter(const list_inputs& /*inputs*/, PVRush::PVNraw const& /*nraw*/) const
	{
		return {};
	}
	virtual PVWidgets::PVExporterWidgetInterface*
	create_exporter_widget(const list_inputs& /*inputs*/, PVRush::PVNraw const& /*nraw*/) const
	{
		return nullptr;
	}
	virtual QString get_exporter_filter_string(const list_inputs& /*inputs*/) const { return {}; }

	virtual QString name() const = 0;
	virtual QString human_name() const = 0;
	virtual QString human_name_serialize() const = 0;
	virtual QString internal_name() const = 0;
	// Warning: the "human name" of an input must be *unique* accross all the possible inputs
	virtual QString human_name_of_input(PVInputDescription_p in) const { return in->human_name(); };
	virtual QString menu_input_name() const = 0;
	virtual QKeySequence menu_shortcut() const { return QKeySequence(); }
	virtual QString tab_name_of_inputs(list_inputs const& in) const = 0;
	virtual bool get_custom_formats(PVInputDescription_p in, hash_formats& formats) const = 0;
	virtual PVInputDescription_p serialize_read(PVCore::PVSerializeObject& so) = 0;
	virtual QIcon icon() const { return QIcon(); }
	virtual QCursor cursor() const { return QCursor(); }

	virtual void save_input_to_qsettings(const PVInputDescription& input_descr,
	                                     QSettings& settings) = 0;
	virtual PVInputDescription_p load_input_from_string(std::vector<std::string> const&) = 0;
	virtual std::vector<std::string> load_input_descr_from_qsettings(QSettings const& v) = 0;

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
		Q_EMIT edit_format_signal(path, parent);
	}

	void edit_format(QDomDocument& doc, QWidget* parent) const
	{
		Q_EMIT edit_format_signal(doc, parent);
	}

	void connect_parent(QObject const* parent) const
	{
		disconnect((QObject*)this, SIGNAL(edit_format_signal(QString const&, QWidget*)), parent,
		           SLOT(edit_format_Slot(QString const&, QWidget*)));
		disconnect((QObject*)this, SIGNAL(edit_format_signal(QDomDocument&, QWidget*)), parent,
		           SLOT(edit_format_Slot(QDomDocument&, QWidget*)));
		connect((QObject*)this, SIGNAL(edit_format_signal(QString const&, QWidget*)), parent,
		        SLOT(edit_format_Slot(QString const&, QWidget*)));
		connect((QObject*)this, SIGNAL(edit_format_signal(QDomDocument&, QWidget*)), parent,
		        SLOT(edit_format_Slot(QDomDocument&, QWidget*)));
	}
  Q_SIGNALS:
	void edit_format_signal(QString const& path, QWidget* parent) const;
	void edit_format_signal(QDomDocument& doc, QWidget* parent) const;
};

template <typename T>
class PVInputTypeDesc : public PVInputType
{
	PVInputDescription_p serialize_read(PVCore::PVSerializeObject& so) override
	{
		return T::serialize_read(so);
	}

	void save_input_to_qsettings(const PVInputDescription& input_descr,
	                             QSettings& settings) override
	{
		input_descr.save_to_qsettings(settings);
	}

	PVInputDescription_p load_input_from_string(std::vector<std::string> const& v) override
	{
		return T::load_from_string(v);
	}

	std::vector<std::string> load_input_descr_from_qsettings(QSettings const& v) override
	{
		return T::desc_from_qsetting(v);
	}
};

typedef PVInputType::p_type PVInputType_p;
} // namespace PVRush

#endif
