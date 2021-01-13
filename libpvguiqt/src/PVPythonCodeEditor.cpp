/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2020
 */

#include <pvguiqt/PVPythonCodeEditor.h>

#include <QMimeData>

#include <KF5/KSyntaxHighlighting/syntaxhighlighter.h>
#include <KF5/KSyntaxHighlighting/repository.h>
#include <KF5/KSyntaxHighlighting/definition.h>
#include <KF5/KSyntaxHighlighting/theme.h>

const char* PVGuiQt::PVPythonCodeEditor::PVPythonCodeEditor::_theme_types_name[] = {"ayu Light", "ayu Dark"};

PVGuiQt::PVPythonCodeEditor::PVPythonCodeEditor(EThemeType theme_type, QWidget* parent /* = nullptr */)
    : QTextEdit(parent)
{
    auto repository = new KSyntaxHighlighting::Repository;
	auto highlighter = new KSyntaxHighlighting::SyntaxHighlighter(document());

    // Font
    QFont font = document()->defaultFont();
	font.setFamily("Monospace");
	font.setPointSizeF(10.5);
	font.setStyleStrategy(QFont::PreferAntialias);
	document()->setDefaultFont(font);

    // Language defintion
    highlighter->setDefinition(repository->definitionForName("Python"));

    // Theme
    const auto& theme = repository->theme(_theme_types_name[(size_t) theme_type]);
    highlighter->setTheme(theme);
    setStyleSheet(QString("QTextEdit { background-color : %1; }").arg(QColor(theme.editorColor(KSyntaxHighlighting::Theme::EditorColorRole::BackgroundColor)).name()));
}

void PVGuiQt::PVPythonCodeEditor::insertFromMimeData(const QMimeData* source)
{
    // Force pasting plain text instead of rich text
    QTextEdit::insertPlainText(source->text());
}