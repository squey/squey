#include <pvkernel/filter/PVFieldsFilter.h>
#include <picviz/PVLayerFilter.h>

#include <PVAxisTagHelp.h>

PVInspector::PVAxisTagHelp::PVAxisTagHelp(QStringList sel_tags, QWidget* parent):
	QDialog(parent)
{
	setupUi(this);
	setWindowTitle(tr("Axis tag help"));

	// Fill the axis tag table
	_table_tags->setColumnCount(4);
	_table_tags->setHorizontalHeaderLabels(QStringList() << tr("Tag name") << tr("Description") << tr("Filters that uses it") << tr("Splitters that provide it"));

	// Get layer-filters tags
	Picviz::PVLayerFilterListTags const& tags = LIB_CLASS(Picviz::PVLayerFilter)::get().get_tags();

	// Get splitters tags
	PVFilter::PVFieldsSplitterListTags const& sp_tags = LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_tags();

	// Create a hash of {tag-name->tag-object} from the splitters tags
	QHash<QString, PVFilter::PVFieldsSplitterTag const*> hash_sp_tags;
	PVFilter::PVFieldsSplitterListTags::const_iterator it_sp;
	for (it_sp = sp_tags.begin(); it_sp != sp_tags.end(); it_sp++) {
		hash_sp_tags[it_sp->name()] = &(*it_sp);
	}

	_table_tags->setRowCount(tags.size());
	Picviz::PVLayerFilterListTags::const_iterator it;
	int row = 0;
	for (it = tags.begin(); it != tags.end(); it++) {
		Picviz::PVLayerFilterTag const& tag = *it;
		QString const& tag_name = tag.name();
		_table_tags->setItem(row, 0, new QTableWidgetItem(tag.name()));
		_table_tags->setItem(row, 1, new QTableWidgetItem(tag.desc()));

		// Create a string with the list of the layer filters that declared this tag
		QString used_by = tag_to_classes_name(tag);
		_table_tags->setItem(row, 2, new QTableWidgetItem(used_by));
		if (hash_sp_tags.contains(tag_name)) {
			_table_tags->setItem(row, 3, new QTableWidgetItem(tag_to_classes_name(*(hash_sp_tags[tag_name]))));
			hash_sp_tags.remove(tag_name);
		}
		row++;
	}

	// Add the remaining splitters tags
	_table_tags->setRowCount(row+hash_sp_tags.size()-1);
	QHash<QString, PVFilter::PVFieldsSplitterTag const*>::iterator it_hash_sp;
	for (it_hash_sp = hash_sp_tags.begin(); it_hash_sp != hash_sp_tags.end(); it_hash_sp++) {
		PVFilter::PVFieldsSplitterTag const& tag = *(it_hash_sp.value());
		_table_tags->setItem(row, 0, new QTableWidgetItem(tag.name()));
		_table_tags->setItem(row, 1, new QTableWidgetItem(tag.desc()));
		_table_tags->setItem(row, 3, new QTableWidgetItem(tag_to_classes_name(tag)));
		row++;
	}
	_table_tags->resizeColumnsToContents();
}
