// clang-format off
#ifndef QABSTRACTSLIDER64_H
#define QABSTRACTSLIDER64_H

#include <QtWidgets/qwidget.h>

QT_BEGIN_HEADER

QT_BEGIN_NAMESPACE

QT_MODULE(Gui)

class QAbstractSlider64Private;

class Q_GUI_EXPORT QAbstractSlider64 : public QWidget
{
    Q_OBJECT

    Q_PROPERTY(qint64 minimum READ minimum WRITE setMinimum)
    Q_PROPERTY(qint64 maximum READ maximum WRITE setMaximum)
    Q_PROPERTY(qint64 singleStep READ singleStep WRITE setSingleStep)
    Q_PROPERTY(qint64 pageStep READ pageStep WRITE setPageStep)
    Q_PROPERTY(qint64 value READ value WRITE setValue NOTIFY valueChanged USER true)
    Q_PROPERTY(qint64 sliderPosition READ sliderPosition WRITE setSliderPosition NOTIFY sliderMoved)
    Q_PROPERTY(bool tracking READ hasTracking WRITE setTracking)
    Q_PROPERTY(Qt::Orientation orientation READ orientation WRITE setOrientation)
    Q_PROPERTY(bool invertedAppearance READ invertedAppearance WRITE setInvertedAppearance)
    Q_PROPERTY(bool invertedControls READ invertedControls WRITE setInvertedControls)
    Q_PROPERTY(bool sliderDown READ isSliderDown WRITE setSliderDown DESIGNABLE false)

public:
    explicit QAbstractSlider64(QWidget *parent=0);
    ~QAbstractSlider64();

    Qt::Orientation orientation() const;

    void setMinimum(qint64);
    qint64 minimum() const;

    void setMaximum(qint64);
    qint64 maximum() const;

    void setRange(qint64 min, qint64 max);

    void setSingleStep(qint64);
    qint64 singleStep() const;

    void setPageStep(qint64);
    qint64 pageStep() const;

    void setTracking(bool enable);
    bool hasTracking() const;

    void setSliderDown(bool);
    bool isSliderDown() const;

    void setSliderPosition(qint64);
    qint64 sliderPosition() const;

    void setInvertedAppearance(bool);
    bool invertedAppearance() const;

    void setInvertedControls(bool);
    bool invertedControls() const;

    enum SliderAction {
        SliderNoAction,
        SliderSingleStepAdd,
        SliderSingleStepSub,
        SliderPageStepAdd,
        SliderPageStepSub,
        SliderToMinimum,
        SliderToMaximum,
        SliderMove
    };

    qint64 value() const;

    void triggerAction(SliderAction action);

public Q_SLOTS:
    void setValue(qint64);
    void setOrientation(Qt::Orientation);

Q_SIGNALS:
    void valueChanged(qint64 value);

    void sliderPressed();
    void sliderMoved(qint64 position);
    void sliderReleased();

    void rangeChanged(qint64 min, qint64 max);

    void actionTriggered(int action);

protected:
    bool event(QEvent *e);

    void setRepeatAction(SliderAction action, int thresholdTime = 500, int repeatTime = 50);
    SliderAction repeatAction() const;

    enum SliderChange {
        SliderRangeChange,
        SliderOrientationChange,
        SliderStepsChange,
        SliderValueChange
    };
    virtual void sliderChange(SliderChange change);

    void keyPressEvent(QKeyEvent *ev);
    void timerEvent(QTimerEvent *);
#ifndef QT_NO_WHEELEVENT
    void wheelEvent(QWheelEvent *e);
#endif
    void changeEvent(QEvent *e);

#ifdef QT3_SUPPORT
public:
    inline QT3_SUPPORT qint64 minValue() const { return minimum(); }
    inline QT3_SUPPORT qint64 maxValue() const { return maximum(); }
    inline QT3_SUPPORT qint64 lineStep() const { return singleStep(); }
    inline QT3_SUPPORT void setMinValue(qint64 v) { setMinimum(v); }
    inline QT3_SUPPORT void setMaxValue(qint64 v) { setMaximum(v); }
    inline QT3_SUPPORT void setLineStep(qint64 v) { setSingleStep(v); }
    inline QT3_SUPPORT void setSteps(qint64 single, qint64 page) { setSingleStep(single); setPageStep(page); }
    inline QT3_SUPPORT void addPage() { triggerAction(SliderPageStepAdd); }
    inline QT3_SUPPORT void subtractPage() { triggerAction(SliderPageStepSub); }
    inline QT3_SUPPORT void addLine() { triggerAction(SliderSingleStepAdd); }
    inline QT3_SUPPORT void subtractLine() { triggerAction(SliderSingleStepSub); }
#endif

protected:
    QAbstractSlider64(QAbstractSlider64Private &dd, QWidget *parent=0);

private:
    Q_DISABLE_COPY(QAbstractSlider64)
    Q_DECLARE_PRIVATE(QAbstractSlider64)
};

QT_END_NAMESPACE

QT_END_HEADER

#endif // QABSTRACTSLIDER64_H
// clang-format on
