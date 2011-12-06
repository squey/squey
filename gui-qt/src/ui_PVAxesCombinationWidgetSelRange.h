/********************************************************************************
** Form generated from reading UI file 'PVAxesCombinationWidgetSelRange.ui'
**
** Created: Tue Dec 6 14:30:29 2011
**      by: Qt User Interface Compiler version 4.7.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_PVAXESCOMBINATIONWIDGETSELRANGE_H
#define UI_PVAXESCOMBINATIONWIDGETSELRANGE_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QComboBox>
#include <QtGui/QDialog>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QFormLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QSpacerItem>
#include <QtGui/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_PVAxesCombinationWidgetSelRange
{
public:
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout;
    QLabel *label_3;
    QComboBox *_combo_reverse;
    QLabel *label_4;
    QGroupBox *groupBox;
    QFormLayout *formLayout;
    QLabel *label;
    QLineEdit *_edit_min;
    QLabel *label_2;
    QLineEdit *_edit_max;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label_5;
    QComboBox *_combo_values_src;
    QSpacerItem *horizontalSpacer;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label_6;
    QLineEdit *_edit_rate;
    QDialogButtonBox *buttonBox;

    void setupUi(QDialog *PVAxesCombinationWidgetSelRange)
    {
        if (PVAxesCombinationWidgetSelRange->objectName().isEmpty())
            PVAxesCombinationWidgetSelRange->setObjectName(QString::fromUtf8("PVAxesCombinationWidgetSelRange"));
        PVAxesCombinationWidgetSelRange->setWindowModality(Qt::NonModal);
        PVAxesCombinationWidgetSelRange->resize(333, 203);
        PVAxesCombinationWidgetSelRange->setModal(false);
        verticalLayout = new QVBoxLayout(PVAxesCombinationWidgetSelRange);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label_3 = new QLabel(PVAxesCombinationWidgetSelRange);
        label_3->setObjectName(QString::fromUtf8("label_3"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(label_3->sizePolicy().hasHeightForWidth());
        label_3->setSizePolicy(sizePolicy);

        horizontalLayout->addWidget(label_3);

        _combo_reverse = new QComboBox(PVAxesCombinationWidgetSelRange);
        _combo_reverse->setObjectName(QString::fromUtf8("_combo_reverse"));

        horizontalLayout->addWidget(_combo_reverse);

        label_4 = new QLabel(PVAxesCombinationWidgetSelRange);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        sizePolicy.setHeightForWidth(label_4->sizePolicy().hasHeightForWidth());
        label_4->setSizePolicy(sizePolicy);

        horizontalLayout->addWidget(label_4);


        verticalLayout->addLayout(horizontalLayout);

        groupBox = new QGroupBox(PVAxesCombinationWidgetSelRange);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        sizePolicy.setHeightForWidth(groupBox->sizePolicy().hasHeightForWidth());
        groupBox->setSizePolicy(sizePolicy);
        formLayout = new QFormLayout(groupBox);
        formLayout->setObjectName(QString::fromUtf8("formLayout"));
        formLayout->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);
        label = new QLabel(groupBox);
        label->setObjectName(QString::fromUtf8("label"));

        formLayout->setWidget(0, QFormLayout::LabelRole, label);

        _edit_min = new QLineEdit(groupBox);
        _edit_min->setObjectName(QString::fromUtf8("_edit_min"));
        _edit_min->setInputMask(QString::fromUtf8(""));

        formLayout->setWidget(0, QFormLayout::FieldRole, _edit_min);

        label_2 = new QLabel(groupBox);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        formLayout->setWidget(1, QFormLayout::LabelRole, label_2);

        _edit_max = new QLineEdit(groupBox);
        _edit_max->setObjectName(QString::fromUtf8("_edit_max"));
        _edit_max->setInputMask(QString::fromUtf8(""));

        formLayout->setWidget(1, QFormLayout::FieldRole, _edit_max);


        verticalLayout->addWidget(groupBox);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        label_5 = new QLabel(PVAxesCombinationWidgetSelRange);
        label_5->setObjectName(QString::fromUtf8("label_5"));
        QSizePolicy sizePolicy1(QSizePolicy::Maximum, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(label_5->sizePolicy().hasHeightForWidth());
        label_5->setSizePolicy(sizePolicy1);

        horizontalLayout_2->addWidget(label_5);

        _combo_values_src = new QComboBox(PVAxesCombinationWidgetSelRange);
        _combo_values_src->setObjectName(QString::fromUtf8("_combo_values_src"));
        QSizePolicy sizePolicy2(QSizePolicy::Maximum, QSizePolicy::Fixed);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(_combo_values_src->sizePolicy().hasHeightForWidth());
        _combo_values_src->setSizePolicy(sizePolicy2);

        horizontalLayout_2->addWidget(_combo_values_src);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer);


        verticalLayout->addLayout(horizontalLayout_2);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        label_6 = new QLabel(PVAxesCombinationWidgetSelRange);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        horizontalLayout_3->addWidget(label_6);

        _edit_rate = new QLineEdit(PVAxesCombinationWidgetSelRange);
        _edit_rate->setObjectName(QString::fromUtf8("_edit_rate"));

        horizontalLayout_3->addWidget(_edit_rate);


        verticalLayout->addLayout(horizontalLayout_3);

        buttonBox = new QDialogButtonBox(PVAxesCombinationWidgetSelRange);
        buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Cancel|QDialogButtonBox::Ok);

        verticalLayout->addWidget(buttonBox);


        retranslateUi(PVAxesCombinationWidgetSelRange);
        QObject::connect(buttonBox, SIGNAL(accepted()), PVAxesCombinationWidgetSelRange, SLOT(accept()));
        QObject::connect(buttonBox, SIGNAL(rejected()), PVAxesCombinationWidgetSelRange, SLOT(reject()));

        QMetaObject::connectSlotsByName(PVAxesCombinationWidgetSelRange);
    } // setupUi

    void retranslateUi(QDialog *PVAxesCombinationWidgetSelRange)
    {
        PVAxesCombinationWidgetSelRange->setWindowTitle(QApplication::translate("PVAxesCombinationWidgetSelRange", "Select range...", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("PVAxesCombinationWidgetSelRange", "Include axes whose values", 0, QApplication::UnicodeUTF8));
        _combo_reverse->clear();
        _combo_reverse->insertItems(0, QStringList()
         << QApplication::translate("PVAxesCombinationWidgetSelRange", "are", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("PVAxesCombinationWidgetSelRange", "are not", 0, QApplication::UnicodeUTF8)
        );
        label_4->setText(QApplication::translate("PVAxesCombinationWidgetSelRange", "in this range:", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("PVAxesCombinationWidgetSelRange", "Range", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("PVAxesCombinationWidgetSelRange", "Minimum:", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("PVAxesCombinationWidgetSelRange", "Maximum:", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("PVAxesCombinationWidgetSelRange", "using values from:", 0, QApplication::UnicodeUTF8));
        _combo_values_src->clear();
        _combo_values_src->insertItems(0, QStringList()
         << QApplication::translate("PVAxesCombinationWidgetSelRange", "plotting", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("PVAxesCombinationWidgetSelRange", "mapping", 0, QApplication::UnicodeUTF8)
        );
        label_6->setText(QApplication::translate("PVAxesCombinationWidgetSelRange", "Percentage of matching values:", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class PVAxesCombinationWidgetSelRange: public Ui_PVAxesCombinationWidgetSelRange {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_PVAXESCOMBINATIONWIDGETSELRANGE_H
