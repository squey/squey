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

#ifndef LIBPVKERNEL_PLUGINS_INPUT_TYPES_COMMON_PVPARAMSWIDGET_H_
#define LIBPVKERNEL_PLUGINS_INPUT_TYPES_COMMON_PVPARAMSWIDGET_H_

#include <ui_widget_params.h>

#include <pvkernel/widgets/PVPresetsWidget.h>

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVCSVExporter.h>
#include <pvkernel/core/PVProgressBox.h>

#include <pvkernel/widgets/PVFileDialog.h>
#include <pvkernel/widgets/PVExportDlg.h>
#include <pvkernel/widgets/PVCSVExporterWidget.h>
#include <pvkernel/rush/PVCSVExporter.h>

#include <QTextStream>

namespace PVWidgets
{
class PVQueryBuilder;
}

namespace PVRush
{

class PVInputType;

/**
 * This class is the base class of PVParamsWidget and is not intended to be used directly
 * or even derived except by PVParamsWidget.
 *
 * It contains all the Presets logic and some common messages raised to the user.
 *
 * Please, refer to PVParamsWidget documentation to subclass it correctly.
 */
class PVParamsWidgetBase : public QDialog, protected Ui::WidgetParams
{
	Q_OBJECT;

  public:
	PVParamsWidgetBase(PVInputType const* in_t,
	                   PVRush::hash_formats const& formats,
	                   QWidget* parent);
	virtual ~PVParamsWidgetBase(){};

  public:
	/** Check if the connection to the server is successful
	 *
	 * @param error Store any occured error if provided
	 *
	 * @return true if successfully connected to the server, false otherwise.
	 */
	virtual bool check_connection(std::string* error = nullptr) = 0;

	/** Get the query result count
	 *
	 * @param error Store any occured error if provided
	 *
	 * @return the result count returned by the query.
	 *         0 if an error occured
	 */
	virtual size_t query_result_count(std::string* error = nullptr) = 0;

	QString get_format_path() const { return _format_path->text(); }

	const QDomDocument& get_custom_format() const { return _custom_format; }
	bool is_format_custom() const { return _custom_format_radio->isChecked(); }

  protected:
	QString get_query_type() const;
	void set_query_type(QString const& query_type);

  protected:
	virtual QString get_export_filters();

  protected Q_SLOTS:
	virtual void export_slot() = 0;
	virtual void query_type_changed_slot() = 0;

	virtual void preset_new_slot(const QString& name) = 0;
	virtual void preset_load_slot() = 0;
	virtual void preset_save_slot() = 0;
	virtual void preset_remove_slot() = 0;
	virtual void load_preset(unsigned int id) = 0;

	virtual void check_connection_slot();
	void query_result_count_slot();
	void load_format();
	void edit_existing_format();
	virtual void edit_custom_format();

  Q_SIGNALS:
	// A bit hacky: this is to be able to call PVParamsWidget::set_info virtual pure function
	// from PVParamsWidget constructor
	void load_preset_deferred(unsigned int);

  protected:
	PVWidgets::PVPresetsWidget* _presets_widget;
	QSettings _settings;
	int64_t _last_load_preset;
	PVInputType const* _in_t;
	PVWidgets::PVQueryBuilder* _querybuilder = nullptr;
	QDomDocument _custom_format;
};

template <typename Input, typename Presets, typename Infos, typename Query>
class PVParamsWidget : public PVParamsWidgetBase
{

  public:
	PVParamsWidget(PVInputType const* in_t, PVRush::hash_formats const& formats, QWidget* parent)
	    : PVParamsWidgetBase(in_t, formats, parent)
	{
		populate_presets();
		load_last_preset();
	}

	~PVParamsWidget()
	{
		if (_last_load_preset != -1) {
			_settings.setValue("last_preset", (typename Presets::id_t)_last_load_preset);
		}
	}

  public:
	/** Fetch the needed data from server in order to initalize the widget
	 *  For eg, retrieving the list of databases to fill a combo box.
	 *
	 * @param infos An Infos object
	 *
	 * @return true on success, false otherwise
	 */
	virtual bool fetch_server_data(const Infos& infos) = 0;

	/** Export the query result
	 *
	 * @param compressor The output compressor
	 * @param sep The CSV separator character
	 * @param quote The CSV quote character
	 * @param header Specify if a header should be exporter as well
	 * @param pbox A reference to the progress bar of the export dialog
	 * @param error Store any occured error if provided
	 *
	 * @return the result count returned by the query.
	 *         0 if an error occured
	 */
	virtual void export_query_result(PVCore::PVStreamingCompressor& compressor,
	                                 const std::string& sep,
	                                 const std::string& quote,
	                                 bool header,
	                                 PVCore::PVProgressBox& pbox,
	                                 std::string* error = nullptr) = 0;

  public:
	/** Get the query string in a form meant to be executed by the server
	 *
	 * @param error Store any occured error if provided
	 *
	 * @return The query string to be executed by the server
	 */
	virtual QString get_server_query(std::string* error = nullptr) const = 0;

	/** Get the query string in a form meant to be serialized/deserialized
	 *
	 * @param error Store any occured error if provided
	 *
	 * @return The query string to be serialized
	 */
	virtual QString get_serialize_query() const = 0;

  public:
	/** Initialize the query widget from a serialized query
	 *
	 * @param error Store any occured error if provided
	 *
	 * @return The query string to be serialized
	 */
	virtual void set_query(QString const& query) = 0;

	/** Return a Query object
	 *
	 * @param error Store any occured error if provided
	 *
	 * @return The Query object
	 */
	Query get_query(std::string* error = nullptr) const
	{
		Query query(get_infos(), get_server_query(error), get_query_type());

		return query;
	}

	/** Initialize the input widget from an Infos object
	 *
	 * @param error Store any occured error if provided
	 */
	virtual bool set_infos(Infos const& infos)
	{
		_txt_host->setText(infos.get_host());
		_port_sb->setValue(infos.get_port());
		_format_path->setText(infos.get_format());
		_existing_format_radio->setChecked(not infos.is_format_custom());

		_auth_enabled_cb->setChecked(infos.get_login().isEmpty() == false);
		_login_txt->setText(infos.get_login());
		_passwd_txt->setText(infos.get_password());

		return true;
	}

	/** Return an Infos object
	 *
	 * @param error Store any occured error if provided
	 *
	 * @return The Query object
	 */
	virtual Infos get_infos() const
	{
		Infos infos;

		infos.set_host(_txt_host->text());
		infos.set_port(_port_sb->value());
		infos.set_format(_format_path->text());
		infos.set_custom_format(_custom_format_radio->isChecked());
		if (_auth_enabled_cb->isChecked()) {
			infos.set_login(_login_txt->text());
			infos.set_password(_passwd_txt->text());
		}

		return infos;
	}

  protected:
	void export_slot() override
	{
		std::string error;
		get_query(&error);

		if (error.empty()) {
			// FileDialog for option selection and file to write
			PVWidgets::PVExportDlg export_dlg;

			QFile file;
			QString filename;
			// Ask for file until a valid name is given or the action is aborted
			while (true) {
				int res = export_dlg.exec();
				filename = export_dlg.selectedFiles()[0];
				if (filename.isEmpty() || res == QDialog::Rejected) {
					return;
				}

				file.setFileName(filename);
				if (file.open(QIODevice::WriteOnly)) {

					break;
				}

				// Error case
				QMessageBox::critical(&export_dlg, tr("Error while exporting the selection"),
				                      tr("Can not create the file \"%1\"").arg(filename));
			}

			// Export query
			const PVRush::PVCSVExporter& exporter =
			    dynamic_cast<PVRush::PVCSVExporter&>(export_dlg.exporter_widget()->exporter());
			std::string sep = exporter.get_sep_char(), quote = exporter.get_quote_char();
			bool header = exporter.get_export_header();
			PVCore::PVStreamingCompressor compressor(filename.toStdString());
			PVCore::PVProgressBox::progress(
			    [&](PVCore::PVProgressBox& pbox) {
				    this->export_query_result(compressor, sep, quote, header, pbox, &error);
			    },
			    "Exporting request result...", this);
			compressor.wait_finished();
		} else {
			QMessageBox::critical(
			    (QWidget*)QObject::parent(), tr("Export failed"),
			    tr("Export failed with the following error:\n\n%1").arg(QString(error.c_str())));
		}
	}

	void load_preset(unsigned int id) override
	{
		Infos infos;
		QString query;
		QString query_type;
		bool ret = Presets::get().get(id, infos, query, query_type);

		if (!ret) {
			// Maybe the user modified the settings by hand...
			QMessageBox::critical(
			    this, "Error while loading preset...",
			    QString("Preset %1 could not be loaded."
			            "Maybe it has been modified and/or deleted by another application."
			            "The list of available presets will be refreshed.")
			        .arg(_presets_widget->get_current_preset_name()),
			    QMessageBox::Ok);
			populate_presets();
			return;
		}

		fetch_server_data(infos);

		if (!set_infos(infos)) {
			QMessageBox::warning(this, "Error while loading preset..",
			                     "Error while loading preset..", QMessageBox::Ok);
			return;
		}

		_query_type_cb->setCurrentIndex(_query_type_cb->findText(query_type));
		set_query_type(query_type);

		set_query(query);

		_last_load_preset = id;
		_presets_widget->select_preset(id - 1);
	}

	void preset_new_slot(const QString& name) override
	{
		QString query = get_serialize_query();
		QString query_type = get_query_type();

		// Set the new presets
		// ignore returned value
		Presets::get().add(name, get_infos(), query, query_type);
	}

	void preset_load_slot() override
	{
		typename Presets::id_t id = get_current_preset_id();
		load_preset(id);
	}

	void preset_save_slot() override
	{
		typename Presets::id_t id = get_current_preset_id();
		QString query = get_serialize_query();
		QString query_type = get_query_type();

		Presets::get().set(id, get_infos(), query, query_type);
	}

	void preset_remove_slot() override
	{
		typename Presets::id_t id = get_current_preset_id();
		Presets::get().rm(id);
	}

	void populate_presets()
	{
		_presets_widget->clear_presets();

		// List presets
		for (const auto& preset : Presets::get().list_id_names()) {
			_presets_widget->add_preset(preset.second, preset.first);
		}
	}

	typename Presets::id_t get_current_preset_id()
	{
		// This assume that an existing preset has been selected !
		// assert(!_presets_widget->is_preset_txt_new());
		return _presets_widget->get_preset_data().toUInt();
	}

  private:
	void load_last_preset()
	{
		_last_load_preset = -1;
		_settings.beginGroup(QString(Presets::PV_SETTINGS_INPUT) + "ui");
		if (_settings.contains("last_preset")) {
			typename Presets::id_t id = _settings.value("last_preset").toUInt();
			Q_EMIT PVParamsWidgetBase::load_preset_deferred((unsigned int)id);
		} else {
			// Load the first preset if any
			if (_presets_widget->get_preset_count() > 0) {
				Q_EMIT PVParamsWidgetBase::load_preset_deferred(
				    (unsigned int)_presets_widget->get_preset_data(0).toUInt());
			}
		}
	}
};

} // namespace PVRush

#endif // LIBPVKERNEL_PLUGINS_INPUT_TYPES_COMMON_PVPARAMSWIDGET_H_
