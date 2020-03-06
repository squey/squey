#include <pvkernel/widgets/PVFileDialog.h>

#include <QGroupBox>
#include <QListWidget>
#include <QLineEdit>
#include <QGridLayout>
#include <QPushButton>

namespace PVWidgets
{

class PVMultipleFileDialog : public PVFileDialog
{
  public:
	PVMultipleFileDialog(QWidget* parent = nullptr, Qt::WindowFlags flags = 0)
	    : PVFileDialog(parent, flags)
	{
	}

	PVMultipleFileDialog(QWidget* parent = nullptr,
	                     const QString& caption = QString(),
	                     const QString& directory = QString(),
	                     const QString& filter = QString())
	    : PVFileDialog(parent, caption, directory, filter), _files_list(new QListWidget())
	{
		setFileMode(QFileDialog::ExistingFiles);
		_files_list->setDragDropMode(QAbstractItemView::InternalMove);
		_files_list->setSelectionMode(QAbstractItemView::ExtendedSelection);

		QGridLayout* main_layout = static_cast<QGridLayout*>(layout());

		_open_button = new QPushButton("&Open");
		connect(_open_button, &QPushButton::clicked, this, [this]() {
			setResult(QDialog::Accepted);
			accept();
		});
		_open_button->setIcon(style()->standardIcon(QStyle::SP_DialogOpenButton));
		_open_button->setEnabled(false);

		const QList<QPushButton*>& btns = findChildren<QPushButton*>();
		for (QPushButton* b : btns) {
			if (b->text().toLower().contains("open")) {
				_add_button = b;
				_add_button->setIcon(style()->standardIcon(QStyle::SP_FileDialogNewFolder));
				setLabelText(QFileDialog::Accept, tr("&Add"));
			} else if (b->text().toLower().contains("cancel")) {
				_remove_button = b;
				setLabelText(QFileDialog::Reject, tr("&Remove"));
				_remove_button->setEnabled(false);
			}
		}
		const QList<QLineEdit*>& edits = findChildren<QLineEdit*>();
		for (QLineEdit* e : edits) {
			if (QString::compare(e->objectName(), "fileNameEdit", Qt::CaseInsensitive) == 0) {
				_filename_edit = e;
			}
		}

		disconnect(_add_button, &QPushButton::clicked, nullptr, nullptr);
		connect(this, &QFileDialog::currentChanged, [&](const QString& path) {
			_open_button->setEnabled(_files_list->count() > 0 or QFileInfo(path).isFile());
			setLabelText(QFileDialog::Accept, tr("&Add"));
		});

		disconnect(_remove_button, &QPushButton::clicked, nullptr, nullptr);
		connect(_remove_button, &QPushButton::clicked, [&]() {
			QList<QListWidgetItem*> items = _files_list->selectedItems();
			for (QListWidgetItem* item : items) {
				_files_list->takeItem(_files_list->row(item));
				delete item;
			}
			_remove_button->setEnabled(_files_list->count());
		});

		connect(_files_list, &QListWidget::currentItemChanged,
		        [this](QListWidgetItem* current, QListWidgetItem*) {
			        _remove_button->setEnabled(current != nullptr);
		        });

		connect(_add_button, &QPushButton::clicked, [this]() {
			for (const QString& path : selectedFiles()) {
				if (_files_list->findItems(path, Qt::MatchExactly).empty()) {
					if (QFileInfo(path).isFile()) {
						_files_list->addItem(path);
					} else {
						setDirectory(path);
					}
				}
			}
		});

		main_layout->addWidget(_files_list, 4, 0, 1, 3);
		main_layout->addWidget(_open_button, 5, 0, 1, 3);
	}

	QList<QUrl> selectedUrls() const
	{
		QList<QUrl> urls;
		for (size_t i = 0; i < _files_list->count(); i++) {
			urls.append(QUrl::fromLocalFile(_files_list->item(i)->text()));
		}
		if (urls.isEmpty() and not _filename_edit->text().isEmpty()) {
			QRegularExpression rx(R"((?:\")([^\"]+)(?:\"))");
			QRegularExpressionMatchIterator it = rx.globalMatch(_filename_edit->text());
			while (it.hasNext()) {
				QRegularExpressionMatch match = it.next();
				if (match.hasMatch()) {
					const QString& path = QDir::cleanPath(directory().absolutePath() +
					                                      QDir::separator() + match.captured(1));
					if (QFileInfo(path).isFile()) {
						urls.append(QUrl::fromLocalFile(path));
					}
				}
			}
			if (urls.isEmpty()) {
				const QString& path = QDir::cleanPath(directory().absolutePath() +
				                                      QDir::separator() + _filename_edit->text());
				if (QFileInfo(path).isFile()) {
					urls.append(QUrl::fromLocalFile(path));
				}
			}
		}
		return urls;
	}

	static QStringList getOpenFileNames(QWidget* parent = nullptr,
	                                    const QString& caption = QString(),
	                                    const QString& dir = QString(),
	                                    const QString& filter = QString(),
	                                    QString* selectedFilter = nullptr,
	                                    Options options = Options())
	{
		const QStringList schemes = QStringList(QStringLiteral("file"));
		const QList<QUrl> selectedUrls = getOpenFileUrls(parent, caption, QUrl::fromLocalFile(dir),
		                                                 filter, selectedFilter, options, schemes);
		QStringList fileNames;
		fileNames.reserve(selectedUrls.size());
		for (const QUrl& url : selectedUrls)
			fileNames << url.toLocalFile();
		return fileNames;
	}

	static QList<QUrl> getOpenFileUrls(QWidget* parent = nullptr,
	                                   const QString& caption = QString(),
	                                   const QUrl& dir = QString(),
	                                   const QString& filter_string = QString(),
	                                   QString* selectedFilter = nullptr,
	                                   Options options = Options(),
	                                   const QStringList& supportedSchemes = QStringList())
	{
		PVMultipleFileDialog dialog(parent, caption, dir.toLocalFile(), filter_string);
		dialog.setOptions(get_options(options));
		dialog.setFileMode(QFileDialog::ExistingFiles);

		customize_for_wsl(dialog);
		dialog.setSupportedSchemes(supportedSchemes);
		if (selectedFilter && !selectedFilter->isEmpty())
			dialog.selectNameFilter(*selectedFilter);
		if (dialog.exec() == QDialog::Accepted) {
			if (selectedFilter)
				*selectedFilter = dialog.selectedNameFilter();
			return dialog.selectedUrls();
		}
		return QList<QUrl>();
	}

  private:
	QPushButton* _add_button;
	QPushButton* _remove_button;
	QPushButton* _open_button;
	QListWidget* _files_list;
	QLineEdit* _filename_edit;
};

} // namespace PVWidgets