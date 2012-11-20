
def main():
    startApplication("picviz-inspector.sh")
    sendEvent("QMouseEvent", waitForObject(":new-solution1.pvi[*] — Picviz Inspector 3.0.0.Import files..._QPushButton_2"), QEvent.MouseButtonPress, 246, 19, Qt.LeftButton, 1, 0)
    sendEvent("QMouseEvent", waitForObject(":new-solution1.pvi[*] — Picviz Inspector 3.0.0.Import files..._QPushButton_2"), QEvent.MouseButtonRelease, 246, 19, Qt.LeftButton, 0, 0)

