//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

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
