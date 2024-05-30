#include <pvguiqt/PVErrorsAndWarnings.h>

#include <squey/PVSource.h>

#include <QVBoxLayout>
#include <QSplitter>
#include <QTextEdit>

size_t PVErrorsAndWarnings::invalid_columns_count(const Squey::PVSource* src)
{
	const PVRush::PVNraw& nraw = src->get_rushnraw();

	size_t invalid_columns_count = 0;
	for (PVCol col(0); col < nraw.column_count(); col++) {
		invalid_columns_count +=
		    bool(nraw.column(col).has_invalid() & pvcop::db::INVALID_TYPE::INVALID);
	}

	return invalid_columns_count;
}

static QString bad_conversions_as_string(const Squey::PVSource* src)
{
	QStringList l;

	auto const& ax = src->get_format().get_axes();
	const PVRush::PVNraw& nraw = src->get_rushnraw();

	// We must cap the number of invalid values and their length because Qt could crash otherwise
	size_t max_total_size = 10000;
	size_t max_values = 1000;

	size_t total_size = 0;
	bool end = false;

	for (size_t row = 0; row < nraw.row_count() and not end; row++) {
		for (PVCol col(0); col < nraw.column_count() and not end; col++) {

			const pvcop::db::array& column = nraw.column(col);
			if (not(column.has_invalid() & pvcop::db::INVALID_TYPE::INVALID)) {
				continue;
			}

			if (not column.is_valid(row)) {
				const std::string invalid_value = column.at(row);

				if (invalid_value == "") {
					continue;
				}

				QString str("row #" + QString::number(row + 1) + " :");
				const QString& axis_name = ax[col].get_name();
				const QString& axis_type = ax[col].get_type();

				str += " " + axis_name + " (" + axis_type + ") : \"" +
				       QString::fromStdString(invalid_value) + "\"";

				l << str;

				total_size += str.size();

				if (max_values-- == 0 or total_size > max_total_size) {
					l << "There are more errors but only the first are shown. You should edit your "
					     "format and specify the proper types.";
					end = true;
				}
			}
		}
	}

	return l.join("\n");
}

PVErrorsAndWarnings::PVErrorsAndWarnings(Squey::PVSource* src, QWidget* invalid_events_dialog)
{
    QVBoxLayout* layout = new QVBoxLayout();

    QSplitter* splitter = new QSplitter(Qt::Vertical);
    //splitter->setChildrenCollapsible(false);
	if (invalid_events_dialog) {
		splitter->addWidget(invalid_events_dialog);
	}
	else {
		splitter->addWidget(new QWidget);
	}
    
    const QString& details = bad_conversions_as_string(src);

    QTextEdit* text_edit = new QTextEdit();
    text_edit->setText(details);
    splitter->addWidget(text_edit);

    layout->addWidget(splitter);

    setLayout(layout);
}