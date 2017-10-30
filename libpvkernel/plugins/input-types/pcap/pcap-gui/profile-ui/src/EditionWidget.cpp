#include "include/EditionWidget.h"
#include "ui_EditionWidget.h"

#include "../include/libpvpcap/shell.h"
#include "../include/libpvpcap.h"

#include <QInputDialog>
#include <QRegularExpression>
#include <QDir>
#include <QMessageBox>
#include <QFile>

#include <algorithm>

EditionWidget::EditionWidget(QWidget* parent) : QWidget(parent), _ui(new Ui::EditionWidget)
{
	_ui->setupUi(this);

	// initialize profile list
	load_profile_list();

	connect(_ui->list, &QListWidget::itemSelectionChanged, this,
	        &EditionWidget::update_button_state);

	connect(_ui->list, &QListWidget::currentItemChanged,
	        [&](QListWidgetItem*, QListWidgetItem* prev) {
		        if (prev)
			        Q_EMIT profile_about_to_change(prev->text());
		    });
}

EditionWidget::~EditionWidget()
{
	delete _ui;
}

QString EditionWidget::selected_profile() const
{
	QString profile;
	if (_ui->list->currentItem()) {
		profile = _ui->list->currentItem()->text();
	}
	return profile;
}

void EditionWidget::load_profile_list()
{
	_ui->list->clear();

	// load user profiles
	for (auto& profile : pvpcap::get_user_profile_list()) {
		_ui->list->addItem(QString::fromStdString(profile));
	}
}

void EditionWidget::update_button_state()
{
	// Enable button once file is selected
	if (_ui->list->currentItem() and _ui->list->currentItem()->isSelected()) {
		_ui->duplicate_button->setEnabled(true);
		_ui->delete_button->setEnabled(true);
		_ui->rename_button->setEnabled(true);

		// profile
		Q_EMIT update_profile(_ui->list->currentItem()->text());
	} else {
		_ui->duplicate_button->setEnabled(false);
		_ui->delete_button->setEnabled(false);
		_ui->rename_button->setEnabled(false);
	}
}

void EditionWidget::on_new_profile_button_clicked()
{
	bool ok;
	QString text = QInputDialog::getText(this, tr("Create new profile..."), tr("Profile name: "),
	                                     QLineEdit::Normal, "", &ok);

	if (ok && !text.isEmpty()) {
		text = text.simplified();
		text.replace(" ", "_");
		// FIXME: remove all non alphanumeric characters
		// text.remove(QRegularExpression("[-`~!@#$%^&*()_—+=|:;<>«»,.?/{}\'\"\[\]\\]"));

		std::string profilename = text.toStdString();
		std::vector<std::string> profile_list = pvpcap::get_user_profile_list();

		// if the name already exists
		int found = 0;
		while (std::find(profile_list.begin(), profile_list.end(), profilename) !=
		       profile_list.end()) {
			found++;
			profilename = text.toStdString() + "_" + std::to_string(found);
		}
		pvpcap::create_profile(profilename);

		// add and select profile
		_ui->list->addItem(QString::fromStdString(profilename));
		_ui->list->setCurrentItem(_ui->list->item(_ui->list->count() - 1));
	}
}

void EditionWidget::on_delete_button_clicked()
{
	auto* item = _ui->list->currentItem();
	QString filename =
	    QString::fromStdString(pvpcap::get_user_profile_path(item->text().toStdString()));

	if (QDir().remove(filename)) {
		_ui->list->removeItemWidget(item);
		delete item;
	} else {
		QMessageBox::warning(this, tr("Delete profile..."), tr("Can't remove the file"));
	}
}

void EditionWidget::on_duplicate_button_clicked()
{
	auto* item = _ui->list->currentItem();
	QString filename =
	    QString::fromStdString(pvpcap::get_user_profile_path(item->text().toStdString()));

	// if the name already exists
	std::string profilename = item->text().toStdString() + "_duplicate";
	std::vector<std::string> profile_list = pvpcap::get_user_profile_list();
	int found = 0;
	while (std::find(profile_list.begin(), profile_list.end(), profilename) != profile_list.end()) {
		found++;
		profilename = item->text().toStdString() + "_duplicate_" + std::to_string(found);
	}

	QString new_filename = QString::fromStdString(pvpcap::get_user_profile_path(profilename));

	if (QFile().copy(filename, new_filename)) {
		// add and select profile
		_ui->list->addItem(QString::fromStdString(profilename));
		_ui->list->setCurrentItem(_ui->list->item(_ui->list->count() - 1));
	} else {
		QMessageBox::warning(this, tr("Duplicate profile..."), tr("Can't copy the file"));
	}
}

void EditionWidget::on_rename_button_clicked()
{
	auto* item = _ui->list->currentItem();

	bool ok;
	QString text = QInputDialog::getText(this, tr("Rename profile..."), tr("Profile name: "),
	                                     QLineEdit::Normal, item->text(), &ok);

	if (ok && !text.isEmpty() && text != item->text()) {
		text = text.simplified();
		text.replace(" ", "_");
		// FIXME: remove all non alphanumeric characters
		// text.remove(QRegularExpression("[-`~!@#$%^&*()_—+=|:;<>«»,.?/{}\'\"\[\]\\]"));

		std::string profilename = text.toStdString();
		std::vector<std::string> profile_list = pvpcap::get_user_profile_list();

		// if the name already exists
		int found = 0;
		while (std::find(profile_list.begin(), profile_list.end(), profilename) !=
		       profile_list.end()) {
			found++;
			profilename = text.toStdString() + "_" + std::to_string(found);
		}

		QString old_filename =
		    QString::fromStdString(pvpcap::get_user_profile_path(item->text().toStdString()));
		QString new_filename = QString::fromStdString(pvpcap::get_user_profile_path(profilename));

		if (QDir().rename(old_filename, new_filename)) {
			item->setText(QString::fromStdString(profilename));
		} else {
			QMessageBox::warning(this, tr("Rename profile..."), tr("Can't rename the file"));
		}
	}
}
