// clang-format off
#ifndef QSCROLLBAR64_H
#define QSCROLLBAR64_H

#include <QtWidgets/qwidget.h>
#include <QtWidgets/qabstractslider64.h>

QT_BEGIN_HEADER

QT_BEGIN_NAMESPACE

QT_MODULE(Gui)

#ifndef QT_NO_SCROLLBAR

class QScrollBar64Private;
class QStyleOptionSlider64;

class Q_GUI_EXPORT QScrollBar64 : public QAbstractSlider64
{
    Q_OBJECT
public:
    explicit QScrollBar64(QWidget *parent=0);
    explicit QScrollBar64(Qt::Orientation, QWidget *parent=0);
    ~QScrollBar64();

    QSize sizeHint() const;
    bool event(QEvent *event);

protected:
    void paintEvent(QPaintEvent *);
    void mousePressEvent(QMouseEvent *);
    void mouseReleaseEvent(QMouseEvent *);
    void mouseMoveEvent(QMouseEvent *);
    void hideEvent(QHideEvent*);
    void sliderChange(SliderChange change);
#ifndef QT_NO_CONTEXTMENU
    void contextMenuEvent(QContextMenuEvent *);
#endif
    void initStyleOption(QStyleOptionSlider64 *option) const;

#ifdef QT3_SUPPORT
public:
    QT3_SUPPORT_CONSTRUCTOR QScrollBar64(QWidget *parent, const char* name);
    QT3_SUPPORT_CONSTRUCTOR QScrollBar64(Qt::Orientation, QWidget *parent, const char* name);
    QT3_SUPPORT_CONSTRUCTOR QScrollBar64(qint64 minValue, qint64 maxValue, qint64 lineStep, qint64 pageStep,
                qint64 value, Qt::Orientation, QWidget *parent=0, const char* name = 0);
    inline QT3_SUPPORT bool draggingSlider() { return isSliderDown(); }
#endif

private:
    friend Q_GUI_EXPORT QStyleOptionSlider64 qt_qscrollbarStyleOption(QScrollBar64 *scrollBar);

    Q_DISABLE_COPY(QScrollBar64)
    Q_DECLARE_PRIVATE(QScrollBar64)
};

#endif // QT_NO_SCROLLBAR

QT_END_NAMESPACE

QT_END_HEADER

#endif // QSCROLLBAR64_H
// clang-format on
