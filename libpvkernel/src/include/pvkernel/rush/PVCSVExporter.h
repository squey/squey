/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2015-2018
 */

#ifndef __INENDI_PVCSVEXPORTER_H__
#define __INENDI_PVCSVEXPORTER_H__

#include <pvkernel/rush/PVExporter.h>
#include <pvkernel/rush/PVUtils.h>

namespace PVRush
{

class PVCSVExporter : public PVRush::PVExporterBase
{
  public:
	static const size_t STEP_COUNT;
	static const std::string default_sep_char;
	static const std::string default_quote_char;

  public:
	using export_func_f = std::function<std::string(
	    PVRow, const PVCore::PVColumnIndexes&, const std::string&, const std::string&)>;

  public:
	PVCSVExporter();
	PVCSVExporter(PVCore::PVColumnIndexes column_indexes,
	              PVRow total_row_count,
	              export_func_f f,
	              const std::string& sep_char = default_sep_char,
	              const std::string& quote_char = default_quote_char);

  public:
	void set_column_indexes(const PVCore::PVColumnIndexes& column_indexes)
	{
		_column_indexes = column_indexes;
	}
	void set_total_row_count(PVRow total_row_count) { _total_row_count = total_row_count; }
	void set_export_func(export_func_f f) { _f = f; }
	void set_sep_char(const std::string& sep_char) { _sep_char = sep_char; }
	void set_quote_char(const std::string& quote_char) { _quote_char = quote_char; }
	void set_header(QStringList column_names)
	{
		PVRush::PVUtils::safe_export(column_names, _sep_char, _quote_char);
		_header = std::string("#") +
		          column_names.join(QString::fromStdString(_sep_char)).toStdString() + "\n";
	}
	void set_export_internal_values(bool export_internal_values)
	{
		_export_internal_values = export_internal_values;
	}
	void set_export_header(bool export_header) { _export_header = export_header; }

	const PVCore::PVColumnIndexes& get_column_indexes() const { return _column_indexes; }
	PVRow get_total_row_count() const { return _total_row_count; }
	const std::string& get_sep_char() const { return _sep_char; }
	const std::string& get_quote_char() const { return _quote_char; }
	const std::string& get_header() const { return _header; }
	bool get_export_internal_values() const { return _export_internal_values; }
	bool get_export_header() const { return _export_header; }

  public:
	void export_rows(const std::string& file_path, const PVCore::PVSelBitField& sel);

  private:
	PVCore::PVColumnIndexes _column_indexes;
	size_t _total_row_count = 0;
	export_func_f _f;
	std::string _sep_char = default_sep_char;
	std::string _quote_char = default_quote_char;
	bool _export_header = true;
	std::string _header;
	bool _export_internal_values = false;
};

} // Inendi

#endif // __INENDI_PVCSVEXPORTER_H__
