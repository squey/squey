
def test_add_correlation():
    activateItem(waitForObjectItem(":new-solution1.pvi[*] — Picviz Inspector 3.0.0_QMenuBar_2", "Correlations"))
    activateItem(waitForObjectItem(":Correlations_PVGuiQt::PVCorrelationMenu", "Create new correlation..."))
    type(waitForObject(":Correlation name:_QLineEdit"), "TestCorrelation1")
    type(waitForObject(":Correlation name:_QLineEdit"), "<Return>")
    sendEvent("QCloseEvent", waitForObject(":Correlations_QDialog"))
    test.compare(findObject(":Correlations.TestCorrelation_QAction").iconText, "TestCorrelation1")

def main():
    
    # Include common functions
    source(findFile("scripts", "common.py"))
    
    # Start picviz
    startApplication("picviz-inspector.sh")
    
    # Import a small source
    import_source(path_files("sources/test-petit.log"))#load_investigation(path_files("roots/test-petit.pvi"))
    
    # Test correlation menu
    test_add_correlation()