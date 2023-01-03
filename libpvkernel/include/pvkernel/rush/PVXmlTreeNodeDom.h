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

#ifndef NODEDOM_H
#define NODEDOM_H

#include <pvkernel/core/PVArgument.h> // for PVArgumentList
#include <pvkernel/filter/PVFieldsFilterParamWidget.h>
#include <pvkernel/core/PVField.h>          // for PVField
#include <pvkernel/core/PVLogger.h>         // for PVLOG_DEBUG
#include <pvkernel/filter/PVFieldsFilter.h> // for PVFieldsBaseFilter, etc

#include <pvbase/types.h> // for PVCol

#include <cassert>     // for assert
#include <memory>      // for __shared_ptr
#include <sys/types.h> // for ssize_t

#include <QDomElement> // for QDomElement, QDomDocument
#include <QHash>       // for QHash
#include <QList>       // for QList
#include <QObject>     // for QObject
#include <QString>     // for QString
#include <QStringList> // for QStringList

class QWidget;

#define PVXmlTreeNodeDom_initXml                                                                   \
	"<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n<!DOCTYPE PVParamXml>\n<param></param>\n"

namespace PVRush
{

class PVXmlTreeNodeDom : public QObject
{
	Q_OBJECT
  public:
	enum class Type { Root, field, RegEx, filter, axis, url, splitter, converter };

	PVXmlTreeNodeDom();
	explicit PVXmlTreeNodeDom(QDomElement const& dom);

	/**
	 * Constructor defining the name for le node.
	 * @param _type
	 * @param _str
	 */
	PVXmlTreeNodeDom(Type _type, const QString& _str, QDomElement& dom, QDomDocument& file);
	void init(Type _type, const QString& _str, QDomElement& dom, QDomDocument& xmlFile_);
	~PVXmlTreeNodeDom() override;

	static PVRush::PVXmlTreeNodeDom* new_format(QDomDocument& file);

	/**
	 * Add a child.
	 * @param child
	 */
	void addChild(PVXmlTreeNodeDom* child);
	/**
	 * Add a child specifying the row index.
	 * @param child
	 * @param row
	 */
	void addChildAt(PVXmlTreeNodeDom* child, int row);

	/**
	 * Remove a child
	 * @param child
	 */
	void removeChild(PVXmlTreeNodeDom* child);

	/**
	 * Return the child from index row.
	 * @param i
	 * @return the child
	 */
	PVXmlTreeNodeDom* getChild(int i);

	/**
	 * Return the list with the children.
	 * @return children list
	 */
	QList<PVXmlTreeNodeDom*> getChildren();

	/**
	 * Return the children count.
	 * @return number
	 */
	int countChildren();

	/**
	 * Return the parent node.
	 * @return parent
	 */
	PVXmlTreeNodeDom* getParent();

	/**
	 * Return index row in the child list of parent.
	 * @return
	 */
	int getRow();

	static void setFromArgumentList(QDomElement& elt,
	                                PVCore::PVArgumentList const& def_args,
	                                PVCore::PVArgumentList const& args);
	void setFromArgumentList(PVCore::PVArgumentList const& args);

	static void toArgumentList(QDomElement& elt,
	                           PVCore::PVArgumentList const& def_args,
	                           PVCore::PVArgumentList& args);
	// static void toArgumentList(QDomElement& elt, PVCore::PVArgumentList& args);
	void toArgumentList(PVCore::PVArgumentList const& default_args, PVCore::PVArgumentList& args);

	bool isEditable()
	{
		if (type == Type::splitter || type == Type::converter || type == Type::filter ||
		    type == Type::url || type == Type::axis || type == Type::RegEx) {
			return true;
		} else {
			return false;
		}
	}

	static void deleteAllAttributes(QDomElement& elt);

	/**
	 * Setup the DomDocument reference.
	 * @param DomDocument
	 */
	void setDoc(QDomDocument& file);

	QString getName();
	void setName(QString nom);

	QString getExpression();
	void setExpression(QString exp);

	int getNbr();
	void setNbr(int nbr);

	void setSplitterPlugin(PVFilter::PVFieldsSplitterParamWidget_p plugin)
	{
		splitterPlugin = plugin;
		splitterPlugin->connect_to_args_changed(this, SLOT(slot_update()));
		splitterPlugin->connect_to_nchilds_changed(this, SLOT(slot_update_number_childs()));
	}

	PVFilter::PVFieldsSplitterParamWidget_p getSplitterPlugin()
	{
		if (!splitterPlugin) {
			if (createSplitterPlugin(xmlDomElement)) {
				splitterPlugin->set_child_count(countChildren());
			}
		}
		return splitterPlugin;
	}

	void setConverterPlugin(PVFilter::PVFieldsConverterParamWidget_p plugin)
	{
		converterPlugin = plugin;
		assert(converterPlugin);
		converterPlugin->connect_to_args_changed(this, SLOT(slot_update()));
	}

	PVFilter::PVFieldsConverterParamWidget_p getConverterPlugin()
	{
		if (!converterPlugin) {
			createConverterPlugin(xmlDomElement);
		}
		return converterPlugin;
	}

	bool createSplitterPlugin(const QDomElement&);
	bool createConverterPlugin(const QDomElement&);

	QDomElement getDom();

	void updateFiltersDataDisplay();

	/**
	 * General attribute setter.
	 * @param name
	 * @param Value
	 */
	void setAttribute(QString name, QString Value, bool flagSaveInXml = true);
	/**
	 * General attribute getter.
	 * @param name
	 * @return
	 */
	QString attribute(QString name, bool flagReadInXml = true);

	QWidget* getSplitterParamWidget()
	{
		PVCore::PVArgumentList args, args_default;
		args_default = getSplitterPlugin()->get_default_argument();
		toArgumentList(args_default, args);
		try {
			getSplitterPlugin()->get_filter()->set_args(args);
		} catch (PVFilter::PVFieldsFilterInvalidArguments const&) {
			// Don't throw an exception for filters containing invalid
			// arguments because the user is currently creating his format
		}
		return getSplitterPlugin()->get_param_widget();
	}

	QWidget* getConverterParamWidget()
	{
		PVCore::PVArgumentList args, args_default;
		args_default = getConverterPlugin()->get_default_argument();
		toArgumentList(args_default, args);
		try {
			getConverterPlugin()->get_filter()->set_args(args);
		} catch (PVFilter::PVFieldsFilterInvalidArguments const&) {
			// Don't throw an exception for filters containing invalid
			// arguments because the user is currently creating his format
		}
		return getConverterPlugin()->get_param_widget();
	}

	void getChildrenFromField(PVCore::PVField const& field);
	void clearFiltersData();

	/**
	 * définie un parent au node.
	 * @param parent
	 */
	void setParent(PVXmlTreeNodeDom* parent);

	void addFilterRacine();

	void deleteFromTree();

	/**
	 * add one field.
	 * @return the axis node that corresponds to that field
	 */
	PVRush::PVXmlTreeNodeDom* addOneField(QString const& name);
	PVRush::PVXmlTreeNodeDom* addOneField(QString const& name, QString const& axis_type);

	/**
	 * Return the type of node in a QString.
	 * @return type
	 */
	QString typeToString();

	/**
	 * return the name of axis regexp or url name of each field.
	 * @return
	 */
	QString getOutName();
	/**
	 * return the node of axis regexp or url name of each field.
	 * @return
	 */
	PVXmlTreeNodeDom* getOutWidget();

	bool isOnRoot;

	QStringList getDataForRegexp() { return _data_for_regexp; }

	PVCol updateFieldLinearId(PVCol id);

	PVCol getFieldLinearId() const { return _field_linear_id; }

	PVXmlTreeNodeDom* getFirstFieldParent();

	bool hasAxisAsChild();

	PVCol setAxesNames(QStringList const& names, PVCol id);

	void setMappingProperties(QString const& mode,
	                          PVCore::PVArgumentList const& def_args,
	                          PVCore::PVArgumentList const& args);
	QString getMappingProperties(PVCore::PVArgumentList const& def_args,
	                             PVCore::PVArgumentList& args);

	void setPlottingProperties(QString const& mode,
	                           PVCore::PVArgumentList const& def_args,
	                           PVCore::PVArgumentList const& args);
	QString getPlottingProperties(PVCore::PVArgumentList const& def_args,
	                              PVCore::PVArgumentList& args);

  private:
	bool isAlreadyExplored;

	/**
	 * method to explore child
	 */
	void explore();

	/**
	 * setup the type
	 * @param nom
	 */
	void setTypeFromString(const QString& nom);

	/**
	 * add 'n' field.
	 * @param n
	 */
	void addField(int n);

	/**
	 * delete 'n' field.
	 * @param n
	 */
	void delField(int n);

	bool isFieldOfUrl();

  private:
	QDomElement getMappingElement();
	QDomElement getPlottingElement();

  public Q_SLOTS:
	void slot_update()
	{
		PVLOG_DEBUG("PVXmlTreeNodeDom slot slot_update()\n");
		if (splitterPlugin) {
			setFromArgumentList(getSplitterPlugin()->get_filter()->get_args());
		} else if (converterPlugin) {
			setFromArgumentList(getConverterPlugin()->get_filter()->get_args());
		}
		Q_EMIT data_changed();
	}

	void slot_update_number_childs()
	{
		assert(splitterPlugin);
		PVLOG_DEBUG("slot_update_number_childs with plugin %x\n", splitterPlugin.get());
		setNbr(splitterPlugin->get_child_count());
		Q_EMIT data_changed();
	}

  Q_SIGNALS:
	void data_changed();

  public:
	Type type;

  private:
	QDomDocument xmlFile;
	QList<PVXmlTreeNodeDom*> children;
	PVXmlTreeNodeDom* parent;

	QDomElement xmlDomElement;
	QString str;

	QHash<QString, QString> otherData;

	PVFilter::PVFieldsSplitterParamWidget_p splitterPlugin;
	PVFilter::PVFieldsConverterParamWidget_p converterPlugin;

	// AG: still the same saturday morning hack
	QStringList _data_for_regexp;

	// Id of a field, when the pipeline of filter is linearised. If this id equals to -1
	// it means that it has children !
	// TODO: list the ids of the children, so that they will be selected !
	PVCol _field_linear_id;
};
} // namespace PVRush
#endif /* NODEDOM_H */
