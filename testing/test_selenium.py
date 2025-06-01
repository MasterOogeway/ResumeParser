from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def test_chrome_selenium():
    # Setup Chrome driver using webdriver-manager
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    
    # Open a website
    driver.get("https://www.google.com")
    
    # Print the page title
    print("Page title is:", driver.title)
    
    # Close the browser
    driver.quit()

if __name__ == "__main__":
    test_chrome_selenium()
