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

#ifndef MONMODELE_H
#define MONMODELE_H
#include <QAbstractItemModel>
#include <QDomDocument>
#include <QDomElement>

#include <QFile>
#include <QMessageBox>
#include <QTextStream>
#include <iostream>
#include <QDebug>
#include <QString>
#include <QSet>

#include <pvkernel/rush/PVXmlTreeNodeDom.h>
#include <PVXmlParamWidget.h>
#include <pvkernel/rush/PVXmlParamParser.h>
#include <pvkernel/filter/PVFieldsFilterParamWidget.h>
#include <squey/PVAxesCombination.h>

#include <memory>

namespace App
{
class PVXmlDomModel : public QAbstractItemModel
{

  public:
	PVXmlDomModel(QWidget* parent = nullptr);
	~PVXmlDomModel() override;

	/*
	 * Toolbar methods
	 */
	PVRush::PVXmlTreeNodeDom* addFilterAfter(QModelIndex& index);
	void applyModification(QModelIndex& index, PVXmlParamWidget* paramBord);

	/*
	* Add items
	*/
	PVRush::PVXmlTreeNodeDom* addAxisIn(const QModelIndex& index);
	PVRush::PVXmlTreeNodeDom* addAxisIn(PVRush::PVXmlTreeNodeDom* parentNode);

	PVRush::PVXmlTreeNodeDom* addSplitter(const QModelIndex& index,
	                                      PVFilter::PVFieldsSplitterParamWidget_p splitterPlugin);
	PVRush::PVXmlTreeNodeDom*
	addConverter(const QModelIndex& index,
	             PVFilter::PVFieldsConverterParamWidget_p converterPlugin);
	void addRegExIn(const QModelIndex& index);
	void addUrlIn(const QModelIndex& index);

	PVRush::PVXmlTreeNodeDom*
	addSplitterWithAxes(const QModelIndex& index,
	                    PVFilter::PVFieldsSplitterParamWidget_p splitterPlugin,
	                    QStringList axesName);

	/*
	 * virtual method from QAbstractItemModel
	 */
	QModelIndex index(int, int, const QModelIndex& = QModelIndex()) const override;
	QModelIndex parent(const QModelIndex&) const override;
	int rowCount(const QModelIndex&) const override;
	int columnCount(const QModelIndex&) const override;
	QVariant data(const QModelIndex&, int) const override;

	// return selectable
	Qt::ItemFlags flags(const QModelIndex& index) const override;

	// drag&drop
	Qt::DropActions supportedDropActions() const override;

	// respond to the View
	bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;

	/**
	 * initialisation of the root.
	 * @param
	 */
	void setRoot(PVRush::PVXmlTreeNodeDom*);

	/**
	 * get the Dom with index.
	 * @param QModelIndex index
	 * @return QDomElement* objetDom
	 */
	QDomElement* getItem(QModelIndex& index);

	/**
	 * get the version of the format
	 * @return
	 */
	QString getVersion() { return xmlRootDom.attribute("version", "0"); }
	void setVersion(QString v) { xmlRootDom.setAttribute("version", v); }

	size_t get_first_line() const { return xmlRootDom.attribute("first_line", "0").toULongLong(); }
	void set_first_line(size_t value) { xmlRootDom.setAttribute("first_line", (qulonglong)value); }

	size_t get_line_count() const { return xmlRootDom.attribute("line_count", "0").toULongLong(); }
	void set_line_count(size_t value)
	{
		if (value)
			xmlRootDom.setAttribute("line_count", (qulonglong)value);
		else
			xmlRootDom.removeAttribute("line_count");
	}

	void set_python_script(const QString& python_script, bool is_path, bool disabled);
	QString get_python_script(bool& is_path, bool& disabled) const;

	/**
	 *
	 * @param section : raw or col index
	 * @param orientation : header vertical or not
	 * @param role : what we are doing
	 * @return something to write on tree header
	 */
	QVariant
	headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;

	bool saveXml(QString xml_file);

	void deleteSelection(QModelIndex const& index);

	void moveDown(const QModelIndex& index);
	void moveUp(const QModelIndex& index);
	QModelIndex selectNext(const QModelIndex& index);

	// open a pcre
	bool openXml(QString file);
	void openXml(QDomDocument& doc);

	bool hasFormatChanged() const;

	// identify multi axis or splitter in a field
	bool trustConfictSplitAxes(const QModelIndex& index);

	// find level count form index to parent
	int countParent(const QModelIndex& index);

	PVRush::PVXmlTreeNodeDom* nodeFromIndex(const QModelIndex& index) const;
	QModelIndex indexOfChild(const QModelIndex& parent, const PVRush::PVXmlTreeNodeDom* node) const;

	QDomElement const& getRootDom() const { return xmlRootDom; }
	QDomElement& getRootDom() { return xmlRootDom; }

	PVRush::PVXmlTreeNodeDom* getRoot() const { return rootNode.get(); }

	void processChildrenWithField(PVCore::PVField const& field);
	void clearFiltersData();
	void updateFieldsLinearId();
	void updateFiltersDataDisplay();
	void setAxesNames(QStringList const& names);
	void updateAxesCombination();

	size_t get_axes_count() const;

	Squey::PVAxesCombination& get_axes_combination() { return _axes_combination; }

  private:
	static void setDefaultAttributesForAxis(QDomElement& elt);
	void setEltMappingPlotting(QDomElement& elt,
	                           QString const& type,
	                           QString const& mode_mapping,
	                           QString const& mode_plotting);

  private:
	std::unique_ptr<PVRush::PVXmlTreeNodeDom> rootNode;

	QString urlXml;
	QDomDocument xmlFile;
	QString _original_xml_content;
	QDomElement xmlRootDom;

	QList<PVRush::PVAxisFormat> _axes;
	Squey::PVAxesCombination _axes_combination;
};
} // namespace App
#endif /* MONMODELE_H */
