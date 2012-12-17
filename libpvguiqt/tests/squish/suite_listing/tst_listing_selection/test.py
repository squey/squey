def test_selection():
    type(waitForObject(":PVListingView_PVGuiQt::PVListingView"), "<Shift>")
    waitForObjectItem(":PVListingView_PVGuiQt::PVListingView", "6/1")
    clickItem(":PVListingView_PVGuiQt::PVListingView", "6/1", 53, 15, 33554432, Qt.LeftButton)
    waitForObjectItem(":PVListingView_PVGuiQt::PVListingView", "19/1")
    clickItem(":PVListingView_PVGuiQt::PVListingView", "19/1", 33, 15, 33554432, Qt.LeftButton)
    type(waitForObject(":PVListingView_PVGuiQt::PVListingView"), "<Control>")
    waitForObjectItem(":PVListingView_PVGuiQt::PVListingView", "27/1")
    clickItem(":PVListingView_PVGuiQt::PVListingView", "27/1", 26, 12, 67108864, Qt.LeftButton)
    type(waitForObject(":PVListingView_PVGuiQt::PVListingView"), "<Return>")
    
    # Verify lines text
    test.vp("VP1")
    
    # Verify lines count
    verticalHeader = waitForObject(":PVListingView.verticalHeader_of_PVListingView_QHeaderView")
    test.verify(verticalHeader.count() == 21)
    
def test_select_all():
    type(waitForObject(":PVListingView_PVGuiQt::PVListingView"), "a")
    verticalHeader = waitForObject(":PVListingView.verticalHeader_of_PVListingView_QHeaderView")
    test.verify(verticalHeader.count() == 1833)

def main():
    # Include common functions
    source(findFile("scripts", "common.py"))
    
    # Start test
    startApplication("Tqt_stats_listing " +INSPECTOR_FILES_DIR+ "/sources/test-petit.log " +INSPECTOR_FILES_DIR+ "/formats/v5/syslog.format")
    setWindowState(waitForObject(":_QMainWindow"), WindowState.Maximize)
    
    test_selection()
    test_select_all()