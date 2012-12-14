def unique_values_test_manual_refresh():
    clickButton(waitForObject(":..._QPushButton"))
    waitFor("object.exists(':....N/A_QLabel')", 20000)
    test.compare(findObject(":....N/A_QLabel").text, "N/A")
    waitFor("object.exists(':....N/A_QLabel_2')", 20000)
    test.compare(findObject(":....N/A_QLabel_2").text, "N/A")
    waitFor("object.exists(':....N/A_QLabel_3')", 20000)
    test.compare(findObject(":....N/A_QLabel_3").text, "N/A")
    waitFor("object.exists(':....N/A_QLabel_4')", 20000)
    test.compare(findObject(":....N/A_QLabel_4").text, "N/A")
    clickButton(waitForObject(":....N/A_QPushButton"))
    clickButton(waitForObject(":....N/A_QPushButton"))
    clickButton(waitForObject(":....N/A_QPushButton"))
    clickButton(waitForObject(":....N/A_QPushButton"))
    waitFor("object.exists(':....265_QLabel')", 20000)
    test.compare(findObject(":....265_QLabel").text, "265")
    waitFor("object.exists(':....1_QLabel')", 20000)
    test.compare(findObject(":....1_QLabel").text, "1")
    waitFor("object.exists(':....20_QLabel')", 20000)
    test.compare(findObject(":....20_QLabel").text, "20")
    waitFor("object.exists(':....1,355_QLabel')", 20000)
    test.compare(findObject(":....1,355_QLabel").text, "1,355")
    
def unique_values_test_auto_refresh():
    clickButton(waitForObject(":..._QPushButton_2"))
    clickButton(waitForObject(":..._QPushButton_3"))
    type(waitForObject(":..._QPushButton"), "a")
    waitFor("object.exists(':....N/A_QLabel')", 20000)
    test.compare(findObject(":....N/A_QLabel").text, "N/A")
    waitFor("object.exists(':....1_QLabel')", 20000)
    test.compare(findObject(":....1_QLabel").text, "1")
    waitFor("object.exists(':....20_QLabel')", 20000)
    test.compare(findObject(":....20_QLabel").text, "20")
    waitFor("object.exists(':....N/A_QLabel_2')", 20000)
    test.compare(findObject(":....N/A_QLabel_2").text, "N/A")
    
def unique_values_test_refresh_all():
    clickButton(waitForObject(":..._QPushButton_2"))
    clickButton(waitForObject(":..._QPushButton_3"))
    type(waitForObject(":..._QPushButton"), "a")
    mouseClick(waitForObject(":unique\nvalues_HeaderViewItem"), 14, 11, 0, Qt.LeftButton)
    waitFor("object.exists(':....265_QLabel')", 20000)
    test.compare(findObject(":....265_QLabel").text, "265")
    waitFor("object.exists(':....1_QLabel')", 20000)
    test.compare(findObject(":....1_QLabel").text, "1")
    waitFor("object.exists(':....20_QLabel')", 20000)
    test.compare(findObject(":....20_QLabel").text, "20")
    waitFor("object.exists(':....1,355_QLabel')", 20000)
    test.compare(findObject(":....1,355_QLabel").text, "1,355")
    
def resize_horizontal_headers():
    dragItemBy(waitForObject(":PVListingView.horizontalHeader_of_PVListingView_QHeaderView"), 98, 7, -26, 0, 1, Qt.LeftButton)
    dragItemBy(waitForObject(":PVListingView.horizontalHeader_of_PVListingView_QHeaderView"), 173, 11, -32, 0, 1, Qt.LeftButton)
    dragItemBy(waitForObject(":PVListingView.horizontalHeader_of_PVListingView_QHeaderView"), 239, 10, 107, 1, 1, Qt.LeftButton)
    
    table_widget = waitForObject(":..._QTableWidget")
    
    headers_width = (74, 68, 207, 100)
    for col in range(len(headers_width)):
        test.verify(table_widget.columnWidth(col) == headers_width[col])
        
def unique_values_change_axes_combination():
    type(waitForObject(":..._QPushButton"), "b")
    mouseClick(waitForObject(":unique\nvalues_HeaderViewItem"), 27, 9, 0, Qt.LeftButton)
    waitFor("object.exists(':....1_QLabel')", 20000)
    test.compare(findObject(":....1_QLabel").text, "1")
    waitFor("object.exists(':....20_QLabel')", 20000)
    test.compare(findObject(":....20_QLabel").text, "20")
    waitFor("object.exists(':....1,355_QLabel')", 20000)
    test.compare(findObject(":....1,355_QLabel").text, "1,355")
    waitFor("object.exists(':....1_QLabel_2')", 20000)
    test.compare(findObject(":....1_QLabel_2").text, "1")
    
def unique_values_show_dialog():
    sendEvent("QMouseEvent", waitForObject(":..._QPushButton_4"), QEvent.MouseButtonPress, 9, 9, Qt.LeftButton, 1, 0)
    sendEvent("QMouseEvent", waitForObject(":..._QPushButton_4"), QEvent.MouseButtonRelease, 9, 9, Qt.LeftButton, 0, 0)
    waitFor("object.exists(':Unique values of axis \\'Machine\\'_PVGuiQt::PVListUniqStringsDlg')", 20000)
    test.compare(findObject(":Unique values of axis 'Machine'_PVGuiQt::PVListUniqStringsDlg").windowTitle, "Unique values of axis 'Machine'")
    clickButton(waitForObject(":..._QPushButton_2"))
    sendEvent("QMouseEvent", waitForObject(":..._QPushButton_5"), QEvent.MouseButtonPress, 10, 8, Qt.LeftButton, 1, 0)
    sendEvent("QMouseEvent", waitForObject(":..._QPushButton_5"), QEvent.MouseButtonRelease, 10, 8, Qt.LeftButton, 0, 0)
    waitFor("object.exists(':Unique values of axis \\'Service\\'_PVGuiQt::PVListUniqStringsDlg')", 20000)
    test.compare(findObject(":Unique values of axis 'Service'_PVGuiQt::PVListUniqStringsDlg").windowTitle, "Unique values of axis 'Service'")
    clickButton(waitForObject(":....Close_QPushButton"))
    sendEvent("QMouseEvent", waitForObject(":..._QPushButton_6"), QEvent.MouseButtonPress, 11, 9, Qt.LeftButton, 1, 0)
    sendEvent("QMouseEvent", waitForObject(":..._QPushButton_6"), QEvent.MouseButtonRelease, 11, 9, Qt.LeftButton, 0, 0)
    waitFor("object.exists(':Unique values of axis \\'Message\\'_PVGuiQt::PVListUniqStringsDlg')", 20000)
    test.compare(findObject(":Unique values of axis 'Message'_PVGuiQt::PVListUniqStringsDlg").windowTitle, "Unique values of axis 'Message'")
    clickButton(waitForObject(":....Close_QPushButton"))
    sendEvent("QMouseEvent", waitForObject(":..._QPushButton_7"), QEvent.MouseButtonPress, 6, 9, Qt.LeftButton, 1, 0)
    sendEvent("QMouseEvent", waitForObject(":..._QPushButton_7"), QEvent.MouseButtonRelease, 6, 9, Qt.LeftButton, 0, 0)
    waitFor("object.exists(':Unique values of axis \\'Machine\\'_PVGuiQt::PVListUniqStringsDlg')", 20000)
    test.compare(findObject(":Unique values of axis 'Machine'_PVGuiQt::PVListUniqStringsDlg").windowTitle, "Unique values of axis 'Machine'")
    clickButton(waitForObject(":....Close_QPushButton"))
    
def scrollbar_sync():
    dragItemBy(waitForObject(":PVListingView.horizontalHeader_of_PVListingView_QHeaderView"), 374, 13, 376, 1, 1, Qt.LeftButton)
    dragItemBy(waitForObject(":PVListingView.horizontalHeader_of_PVListingView_QHeaderView"), 272, 7, 294, 5, 1, Qt.LeftButton)
    dragItemBy(waitForObject(":PVListingView.horizontalHeader_of_PVListingView_QHeaderView"), 68, 10, 331, 0, 1, Qt.LeftButton)
    scrollTo(waitForObject(":PVListingView.horizontalScrollBar_of_PVListingView_QScrollBar"), 3)
    waitFor("object.exists(':..._QScrollBar')", 20000)
    #test.compare(findObject(":..._QScrollBar").value, 3)
    #test.compare(findObject(":..._QScrollBar").sliderPosition, 3)

def main():
    # Include common functions
    source(findFile("scripts", "common.py"))

    # Start test
    startApplication("Tqt_stats_listing " +INSPECTOR_FILES_DIR+ "/sources/test-petit.log " +INSPECTOR_FILES_DIR+ "/formats/v5/syslog.format")
    
    unique_values_test_manual_refresh()
    unique_values_test_auto_refresh()
    unique_values_test_refresh_all()
    
    resize_horizontal_headers()
    unique_values_change_axes_combination()
    unique_values_show_dialog()
    
    scrollbar_sync()