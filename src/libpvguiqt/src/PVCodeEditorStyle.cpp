//
// MIT License
//
// © Squey, 2026
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

#include <pvguiqt/PVCodeEditorStyle.h>

#include <pvguiqt/PVGuiQtSettings.h>

#include <pvkernel/core/PVSettings.h>
#include <pvkernel/core/PVTheme.h>

#include <KF6/KSyntaxHighlighting/KSyntaxHighlighting/definition.h>
#include <KF6/KSyntaxHighlighting/KSyntaxHighlighting/repository.h>
#include <KF6/KSyntaxHighlighting/KSyntaxHighlighting/syntaxhighlighter.h>
#include <KF6/KSyntaxHighlighting/KSyntaxHighlighting/theme.h>

#include <QColor>
#include <QFont>
#include <QTextEdit>

void PVGuiQt::style_code_editor(QTextEdit* editor, QString const& syntax)
{
	// Indexed by PVCore::PVTheme::EColorScheme.
	static const char* const THEME_NAMES[] = {"ayu Light", "ayu Dark"};
	// Reading the definitions takes a while, and every editor shares them.
	static KSyntaxHighlighting::Repository repository;

	auto set_font = [editor]() {
		QFont font = editor->document()->defaultFont();
		font.setFamily(settings::editor_font_family.value());
		font.setPointSizeF(settings::editor_font_size.value());
		font.setStyleStrategy(QFont::PreferAntialias);
		editor->document()->setDefaultFont(font);
	};
	set_font();
	// The editors already open follow the settings too.
	QObject::connect(&PVCore::PVSettings::get(), &PVCore::PVSettings::changed, editor,
	                 [set_font](QString const& key) {
		                 if (key == settings::editor_font_family.key() or
		                     key == settings::editor_font_size.key()) {
			                 set_font();
		                 }
	                 });

	auto* highlighter = new KSyntaxHighlighting::SyntaxHighlighter(editor->document());
	highlighter->setDefinition(repository.definitionForName(syntax));
	const auto& theme = repository.theme(THEME_NAMES[(size_t)PVCore::PVTheme::color_scheme()]);
	highlighter->setTheme(theme);
	editor->setStyleSheet(
	    QString("QTextEdit { background-color : %1; }")
	        .arg(QColor(theme.editorColor(
	                        KSyntaxHighlighting::Theme::EditorColorRole::BackgroundColor))
	                 .name()));
}
