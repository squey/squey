/**
 * @file
 *
 * @copyright (C) Picviz Labs 2015
 */

#ifndef LIBPVKERNEL_PLUGINS_INPUT_TYPES_COMMON_PVPARAMSWIDGET_H_
#define LIBPVKERNEL_PLUGINS_INPUT_TYPES_COMMON_PVPARAMSWIDGET_H_

#include <ui_widget_params.h>

#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/widgets/PVPresetsWidget.h>

#include <QFileDialog>

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
	PVParamsWidgetBase(PVInputType const* in_t, PVRush::hash_formats const& formats, QWidget* parent);
	virtual ~PVParamsWidgetBase() {};

protected:
	virtual bool check_connection(std::string* error = nullptr) = 0;
	virtual size_t query_result_count(std::string* error = nullptr) = 0;

protected:
	QString get_query_type() const;
	void set_query_type(QString const& query_type);

protected slots:
	virtual void export_slot() = 0;
	virtual void query_type_changed_slot() = 0;

	virtual void preset_new_slot(const QString& name) = 0;
	virtual void preset_load_slot() = 0;
	virtual void preset_save_slot() = 0;
	virtual void preset_remove_slot() = 0;
	virtual void load_preset(unsigned int id) = 0;

	void check_connection_slot();
	void query_result_count_slot();

signals:
	// A bit hacky: this is to be able to call PVParamsWidget::set_info virtual pure function
    // from PVParamsWidget constructor
	void load_preset_deferred(unsigned int);

protected:
	PVWidgets::PVPresetsWidget* _presets_widget;
	QSettings _settings;
	int64_t _last_load_preset;
	PVInputType const* _in_t;
	PVWidgets::PVQueryBuilder* _querybuilder = nullptr;
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
			_settings.setValue("last_preset", (typename Presets::id_t) _last_load_preset);
		}
	}

public:
    /** Check if the connection to the server is successful
     *
     * @param error Store any occured error if provided
     *
     * @return true if successfully connected to the server, false otherwise.
     */
	virtual bool check_connection(std::string* error = nullptr) = 0;

    /** Fetch the needed data from server in order to initalize the widget
     *  For eg, retrieving the list of databases to fill a combo box.
     *
     * @param infos An Infos object
     *
     * @return true on success, false otherwise
     */
	virtual bool fetch_server_data(const Infos& infos) = 0;

    /** Get the query result count
     *
     * @param error Store any occured error if provided
     *
     * @return the result count returned by the query.
     *         0 if an error occured
     */
	virtual size_t query_result_count(std::string* error = nullptr) = 0;

    /** Export the query result
     *
     * @param output_stream The output stream
     * @param pbox A reference to the progress bar of the export dialog
     * @param error Store any occured error if provided
     *
     * @return the result count returned by the query.
     *         0 if an error occured
     */
	virtual void export_query_result(QTextStream& output_stream, PVCore::PVProgressBox& pbox, std::string* error = nullptr) = 0;

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
		if (_auth_enabled_cb->isChecked()) {
			infos.set_login(_login_txt->text());
			infos.set_password(_passwd_txt->text());
		}

		return infos;
	}

protected:
	void export_slot()
	{
		std::string error;
		const Query& query = get_query(&error);

		if (error.empty()) {
			QString csv_filename = QFileDialog::getSaveFileName(
				this,
				"Export to...",
				"",
				QString("CSV File (*.csv);;All files (*.*)")
			);

			if (csv_filename.isEmpty() == false) {

				QFile f(csv_filename);
				if (f.open(QIODevice::WriteOnly | QIODevice::Text)) {

					QTextStream output_stream(&f);
					PVCore::PVProgressBox pbox("Exporting request result...");
					PVCore::PVProgressBox::progress([&]() {
						export_query_result(output_stream, pbox, &error);
					}, &pbox);
				}
			}
		}

		if (error.empty() == false) {
			QMessageBox::critical(
				(QWidget*) QObject::parent(),
				tr("Export failed"),
				tr("Export failed with the following error:\n\n%1").arg(QString(error.c_str()))
			);
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
				this,
				"Error while loading preset...",
				QString(
					"Preset %1 could not be loaded."
					"Maybe it has been modified and/or deleted by another application."
					"The list of available presets will be refreshed.").arg(_presets_widget->get_current_preset_name()),
					QMessageBox::Ok
				);
			populate_presets();
			return;
		}

		fetch_server_data(infos);

		if (!set_infos(infos)) {
			QMessageBox::warning(this, "Error while loading preset..", "Error while loading preset..", QMessageBox::Ok);
			return;
		}

		_query_type_cb->setCurrentIndex(_query_type_cb->findText(query_type));
		set_query_type(query_type);

		set_query(query);

		_last_load_preset = id;
		_presets_widget->select_preset(id-1);
	}

	void preset_new_slot(const QString& name) override
	{
		QString query = get_server_query();
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
			_presets_widget->add_preset(preset.second,  preset.first);
		}
	}

	typename Presets::id_t get_current_preset_id()
	{
		// This assume that an existing preset has been selected !
		//assert(!_presets_widget->is_preset_txt_new());
		return _presets_widget->get_preset_data().toUInt();
	}

private:
	void load_last_preset()
	{
		_last_load_preset = -1;
		_settings.beginGroup(QString(Presets::PV_SETTINGS_INPUT) + "ui");
		if (_settings.contains("last_preset")) {
			typename Presets::id_t id = _settings.value("last_preset").toUInt();
			emit PVParamsWidgetBase::load_preset_deferred((unsigned int) id);
		}
		else {
			// Load the first preset if any
			if (_presets_widget->get_preset_count() > 0) {
				emit PVParamsWidgetBase::load_preset_deferred((unsigned int) _presets_widget->get_preset_data(0).toUInt());
			}
		}
	}
};

} // namespace PVRush

#endif // LIBPVKERNEL_PLUGINS_INPUT_TYPES_COMMON_PVPARAMSWIDGET_H_
