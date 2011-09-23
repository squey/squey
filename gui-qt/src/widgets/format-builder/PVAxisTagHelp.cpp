#include <PVAxisTagHelp.h>


PVInspector::PVAxisTagHelp::PVAxisTagHelp(Picviz::PVLayerFilterTag sel_tag, QWidget* parent):
	QDialog(parent)
{
	setupUi(this);
	setWindowTitle(tr("Axis tag help"));

	// Fill the axis tag table
	_table_tags->setColumnCount(3);
	_table_tags->setHorizontalHeaderLabels(QStringList() << tr("Tag name") << tr("Description") << tr("Filters that uses it"));

	Picviz::PVLayerFilterListTags const& tags = LIB_CLASS(Picviz::PVLayerFilter)::get().get_tags();
	_table_tags->setRowCount(tags.size());
	Picviz::PVLayerFilterListTags::const_iterator it;
	int row = 0;
	for (it = tags.begin(); it != tags.end(); it++) {
		Picviz::PVLayerFilterTag const& tag = *it;
		_table_tags->setItem(row, 0, new QTableWidgetItem(tag.name()));
		_table_tags->setItem(row, 1, new QTableWidgetItem(tag.desc()));

		// Create a string with the list of the layer filters that declared this tag
		QString used_by;
		Picviz::PVLayerFilterTag::list_classes const& filters = tag.associated_classes();
		Picviz::PVLayerFilterTag::list_classes::const_iterator it_c;
		for (it_c = filters.begin(); it_c != filters.end(); it_c++) {
			used_by += (*it_c)->registered_name() + "\n";
		}
		_table_tags->setItem(row, 2, new QTableWidgetItem(used_by));
		row++;
	}
	_table_tags->resizeColumnsToContents();
}
