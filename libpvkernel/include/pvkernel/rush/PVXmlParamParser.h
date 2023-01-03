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

#ifndef PVXMLPARAMPARSER_H
#define PVXMLPARAMPARSER_H

#include <QList>
#include <QDomElement>
#include <QString>

#include <pvkernel/rush/PVAxisFormat.h>
#include <pvkernel/rush/PVXmlParamParserData.h>
#include <pvkernel/rush/PVFormat_types.h>

namespace PVRush
{

class PVInvalidFile : public std::runtime_error
{
  public:
	using std::runtime_error::runtime_error;
};

class PVXmlParamParser
{
  public:
	typedef QList<PVXmlParamParserData> list_params;
	typedef std::vector<PVCol> axes_comb_t;
	using fields_mask_t = std::vector<bool>;

  public:
	explicit PVXmlParamParser(QString const& nameFile, bool add_input_column_name = false);
	explicit PVXmlParamParser(QDomElement const& rootNode);
	virtual ~PVXmlParamParser();

  public:
	int setDom(QDomElement const& node, int id = -1);
	QList<PVAxisFormat> const& getAxes() const;
	QList<PVXmlParamParserData> const& getFields() const;
	const fields_mask_t& getFieldsMask() const { return _fields_mask; }
	unsigned int getVersion() const { return format_version; }
	size_t get_first_line() const { return _first_line; }
	size_t get_line_count() const { return _line_count; }
	void dump_filters();
	void clearFiltersData();
	axes_comb_t const& getAxesCombination() const { return _axes_combination; }
	QString get_python_script(bool& as_path, bool& disabled) const;

  private:
	void pushFilter(const QDomElement& elt, int newId);
	void parseFromRootNode(QDomElement const& node);
	void addInputNameColumn(QDomDocument& xml);
	void setAxesCombinationFromRootNode(QDomElement const& node);
	void setAxesCombinationFromString(QString const& str);
	void setLinesRangeFromRootNode(QDomElement const& rootNode);
	void setPythonScriptFromRootNode(const QDomElement&);
	void setPythonScriptFromFile(QString const& python_script, bool as_path, bool disabled);
	static PVAxisFormat::node_args_t
	getMapPlotParameters(QDomElement& elt, QString const& tag, QString& mode);

  private:
	QList<PVXmlParamParserData> fields;
	QList<PVAxisFormat> _axes;
	unsigned int format_version;
	axes_comb_t _axes_combination;
	size_t _first_line;
	size_t _line_count;
	fields_mask_t _fields_mask;
	QString _python_script;
	bool _python_script_is_path;
	bool _python_script_disabled;

	int countChild(QDomElement);
	QString getNodeName(QDomElement);
	QString getNodeRegExp(QDomElement);
	QString getNodeType(QDomElement);
	QString getNodeTypeGrep(QDomElement node);
};
} // namespace PVRush

#endif /* PVXMLPARAMPARSER_H */
