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
		if (_export_rows_index) {
			column_names.insert(0, "row_index");
		}
		PVRush::PVUtils::safe_export(column_names, _sep_char, _quote_char);
		const std::string& column_names_str = column_names.join(QString::fromStdString(_sep_char)).toStdString();
		_header = std::string("#") + column_names_str + std::string("\n");
	}
	void set_export_internal_values(bool export_internal_values)
	{
		_export_internal_values = export_internal_values;
	}
	void set_export_header(bool export_header) { _export_header = export_header; }
	void set_export_rows_index(bool export_rows_index) { _export_rows_index = export_rows_index; }

	const PVCore::PVColumnIndexes& get_column_indexes() const { return _column_indexes; }
	PVRow get_total_row_count() const { return _total_row_count; }
	const std::string& get_sep_char() const { return _sep_char; }
	const std::string& get_quote_char() const { return _quote_char; }
	const std::string& get_header() const { return _header; }
	bool get_export_internal_values() const { return _export_internal_values; }
	bool get_export_header() const { return _export_header; }
	bool get_export_rows_index() const { return _export_rows_index; }

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
	bool _export_rows_index = false;

	size_t _exported_row_count = 0;
};

} // namespace PVRush

#endif // __INENDI_PVCSVEXPORTER_H__
