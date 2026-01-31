"""
Web Interaction Module
Complete toolkit for web browsing, scraping, form filling, and automation
"""

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from typing import List, Dict, Any, Optional
import logging
import time
import json
from urllib.parse import urljoin, urlparse
import re

logger = logging.getLogger(__name__)


class WebInteractor:
    """
    Comprehensive web interaction capabilities:
    - HTTP requests and API calls
    - Web scraping and parsing
    - Browser automation (Selenium)
    - Form filling and submission
    - Authentication handling
    - Screenshot capture
    - JavaScript execution
    """
    
    def __init__(self, headless: bool = True, timeout: int = 30):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.headless = headless
        self.timeout = timeout
        self.driver = None
        
    def _get_driver(self) -> webdriver.Chrome:
        """Initialize and return Chrome WebDriver"""
        if self.driver is None:
            options = Options()
            if self.headless:
                options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            
            self.driver = webdriver.Chrome(options=options)
            self.driver.set_page_load_timeout(self.timeout)
        
        return self.driver
    
    def close_driver(self):
        """Close the WebDriver"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    # ==================== HTTP REQUESTS ====================
    
    def get_request(self, url: str, params: Optional[Dict] = None, 
                    headers: Optional[Dict] = None) -> Dict[str, Any]:
        """Make GET request"""
        try:
            response = self.session.get(url, params=params, headers=headers or {}, timeout=self.timeout)
            response.raise_for_status()
            
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.text,
                "url": response.url,
                "success": True
            }
        except Exception as e:
            logger.error(f"GET request failed: {e}")
            return {"success": False, "error": str(e)}
    
    def post_request(self, url: str, data: Optional[Dict] = None, 
                     json_data: Optional[Dict] = None,
                     headers: Optional[Dict] = None) -> Dict[str, Any]:
        """Make POST request"""
        try:
            response = self.session.post(
                url, 
                data=data, 
                json=json_data, 
                headers=headers or {}, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.text,
                "url": response.url,
                "success": True
            }
        except Exception as e:
            logger.error(f"POST request failed: {e}")
            return {"success": False, "error": str(e)}
    
    def api_call(self, url: str, method: str = "GET", 
                 params: Optional[Dict] = None,
                 data: Optional[Dict] = None,
                 headers: Optional[Dict] = None,
                 auth: Optional[tuple] = None) -> Dict[str, Any]:
        """Make API call with JSON response handling"""
        try:
            response = self.session.request(
                method=method.upper(),
                url=url,
                params=params,
                json=data,
                headers=headers or {},
                auth=auth,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Try to parse JSON
            try:
                json_data = response.json()
            except:
                json_data = None
            
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "json": json_data,
                "text": response.text,
                "success": True
            }
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return {"success": False, "error": str(e)}
    
    # ==================== WEB SCRAPING ====================
    
    def fetch_page_content(self, url: str) -> Dict[str, Any]:
        """Fetch and parse HTML content"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract various elements
            return {
                "url": url,
                "title": soup.title.string if soup.title else "",
                "text": soup.get_text(separator='\n', strip=True),
                "links": [a.get('href') for a in soup.find_all('a', href=True)],
                "images": [img.get('src') for img in soup.find_all('img', src=True)],
                "headings": {
                    f"h{i}": [h.get_text(strip=True) for h in soup.find_all(f'h{i}')]
                    for i in range(1, 7)
                },
                "meta": {
                    meta.get('name', meta.get('property', '')): meta.get('content', '')
                    for meta in soup.find_all('meta') if meta.get('content')
                },
                "success": True
            }
        except Exception as e:
            logger.error(f"Failed to fetch page: {e}")
            return {"success": False, "error": str(e)}
    
    def scrape_table(self, url: str, table_index: int = 0) -> List[List[str]]:
        """Scrape table data from webpage"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            tables = soup.find_all('table')
            if table_index >= len(tables):
                return []
            
            table = tables[table_index]
            data = []
            
            for row in table.find_all('tr'):
                cols = row.find_all(['td', 'th'])
                data.append([col.get_text(strip=True) for col in cols])
            
            return data
        except Exception as e:
            logger.error(f"Failed to scrape table: {e}")
            return []
    
    def extract_links(self, url: str, filter_pattern: Optional[str] = None) -> List[Dict[str, str]]:
        """Extract all links from a page with optional filtering"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            links = []
            for a in soup.find_all('a', href=True):
                href = a.get('href')
                absolute_url = urljoin(url, href)
                
                if filter_pattern and not re.search(filter_pattern, absolute_url):
                    continue
                
                links.append({
                    "text": a.get_text(strip=True),
                    "href": absolute_url,
                    "title": a.get('title', '')
                })
            
            return links
        except Exception as e:
            logger.error(f"Failed to extract links: {e}")
            return []
    
    def extract_structured_data(self, url: str, selectors: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract structured data using CSS selectors
        selectors: {"field_name": "css_selector", ...}
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            data = {}
            for field, selector in selectors.items():
                elements = soup.select(selector)
                if len(elements) == 1:
                    data[field] = elements[0].get_text(strip=True)
                elif len(elements) > 1:
                    data[field] = [el.get_text(strip=True) for el in elements]
                else:
                    data[field] = None
            
            return {"success": True, "data": data}
        except Exception as e:
            logger.error(f"Failed to extract structured data: {e}")
            return {"success": False, "error": str(e)}
    
    # ==================== BROWSER AUTOMATION ====================
    
    def navigate_to(self, url: str) -> Dict[str, Any]:
        """Navigate to URL using Selenium"""
        try:
            driver = self._get_driver()
            driver.get(url)
            
            return {
                "success": True,
                "url": driver.current_url,
                "title": driver.title
            }
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def click_element(self, selector: str, by: str = "css") -> Dict[str, Any]:
        """Click element by selector"""
        try:
            driver = self._get_driver()
            by_method = By.CSS_SELECTOR if by == "css" else By.XPATH
            
            element = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((by_method, selector))
            )
            element.click()
            
            return {"success": True, "message": "Element clicked"}
        except Exception as e:
            logger.error(f"Click failed: {e}")
            return {"success": False, "error": str(e)}
    
    def fill_form_field(self, selector: str, value: str, by: str = "css") -> Dict[str, Any]:
        """Fill form field with value"""
        try:
            driver = self._get_driver()
            by_method = By.CSS_SELECTOR if by == "css" else By.XPATH
            
            element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((by_method, selector))
            )
            element.clear()
            element.send_keys(value)
            
            return {"success": True, "message": "Field filled"}
        except Exception as e:
            logger.error(f"Fill field failed: {e}")
            return {"success": False, "error": str(e)}
    
    def submit_form(self, form_selector: str, form_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Fill and submit form
        form_data: {"field_selector": "value", ...}
        """
        try:
            driver = self._get_driver()
            
            # Fill all fields
            for selector, value in form_data.items():
                result = self.fill_form_field(selector, value)
                if not result["success"]:
                    return result
            
            # Submit form
            form = driver.find_element(By.CSS_SELECTOR, form_selector)
            form.submit()
            
            time.sleep(2)  # Wait for submission
            
            return {
                "success": True,
                "url": driver.current_url,
                "message": "Form submitted"
            }
        except Exception as e:
            logger.error(f"Form submission failed: {e}")
            return {"success": False, "error": str(e)}
    
    def execute_javascript(self, script: str) -> Any:
        """Execute JavaScript in browser context"""
        try:
            driver = self._get_driver()
            result = driver.execute_script(script)
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"JavaScript execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def take_screenshot(self, filepath: str) -> Dict[str, Any]:
        """Take screenshot of current page"""
        try:
            driver = self._get_driver()
            driver.save_screenshot(filepath)
            return {"success": True, "filepath": filepath}
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return {"success": False, "error": str(e)}
    
    def scroll_page(self, direction: str = "down", amount: int = 500) -> Dict[str, Any]:
        """Scroll page in specified direction"""
        try:
            driver = self._get_driver()
            
            if direction == "down":
                driver.execute_script(f"window.scrollBy(0, {amount});")
            elif direction == "up":
                driver.execute_script(f"window.scrollBy(0, -{amount});")
            elif direction == "bottom":
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            elif direction == "top":
                driver.execute_script("window.scrollTo(0, 0);")
            
            return {"success": True, "message": f"Scrolled {direction}"}
        except Exception as e:
            logger.error(f"Scroll failed: {e}")
            return {"success": False, "error": str(e)}
    
    def wait_for_element(self, selector: str, by: str = "css", timeout: int = 10) -> Dict[str, Any]:
        """Wait for element to appear"""
        try:
            driver = self._get_driver()
            by_method = By.CSS_SELECTOR if by == "css" else By.XPATH
            
            element = WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((by_method, selector))
            )
            
            return {
                "success": True,
                "text": element.text,
                "visible": element.is_displayed()
            }
        except TimeoutException:
            return {"success": False, "error": "Element not found within timeout"}
        except Exception as e:
            logger.error(f"Wait failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_page_source(self) -> str:
        """Get current page HTML source"""
        try:
            driver = self._get_driver()
            return driver.page_source
        except Exception as e:
            logger.error(f"Failed to get page source: {e}")
            return ""
    
    def get_cookies(self) -> List[Dict[str, Any]]:
        """Get all cookies from current session"""
        try:
            driver = self._get_driver()
            return driver.get_cookies()
        except Exception as e:
            logger.error(f"Failed to get cookies: {e}")
            return []
    
    def set_cookie(self, cookie: Dict[str, Any]) -> Dict[str, Any]:
        """Set a cookie"""
        try:
            driver = self._get_driver()
            driver.add_cookie(cookie)
            return {"success": True, "message": "Cookie set"}
        except Exception as e:
            logger.error(f"Failed to set cookie: {e}")
            return {"success": False, "error": str(e)}
    
    # ==================== ADVANCED AUTOMATION ====================
    
    def automated_workflow(self, url: str, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute automated workflow with multiple actions
        actions: [
            {"type": "navigate", "url": "..."},
            {"type": "click", "selector": "..."},
            {"type": "fill", "selector": "...", "value": "..."},
            {"type": "wait", "seconds": 2},
            {"type": "screenshot", "filepath": "..."},
            {"type": "execute_js", "script": "..."}
        ]
        """
        try:
            driver = self._get_driver()
            results = []
            
            # Navigate to initial URL
            driver.get(url)
            results.append({"action": "navigate", "success": True, "url": url})
            
            for action in actions:
                action_type = action.get("type")
                
                if action_type == "click":
                    result = self.click_element(action["selector"], action.get("by", "css"))
                
                elif action_type == "fill":
                    result = self.fill_form_field(action["selector"], action["value"], action.get("by", "css"))
                
                elif action_type == "wait":
                    time.sleep(action.get("seconds", 1))
                    result = {"success": True, "message": f"Waited {action.get('seconds', 1)}s"}
                
                elif action_type == "screenshot":
                    result = self.take_screenshot(action["filepath"])
                
                elif action_type == "execute_js":
                    result = self.execute_javascript(action["script"])
                
                elif action_type == "scroll":
                    result = self.scroll_page(action.get("direction", "down"), action.get("amount", 500))
                
                elif action_type == "wait_element":
                    result = self.wait_for_element(action["selector"], action.get("by", "css"), action.get("timeout", 10))
                
                else:
                    result = {"success": False, "error": f"Unknown action type: {action_type}"}
                
                results.append({"action": action_type, **result})
                
                if not result.get("success", False):
                    break  # Stop on first failure
            
            return {
                "success": True,
                "results": results,
                "final_url": driver.current_url
            }
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            return {"success": False, "error": str(e)}
    
    # ==================== DOWNLOAD ====================
    
    def download_file(self, url: str, filepath: str) -> Dict[str, Any]:
        """Download file from URL"""
        try:
            response = self.session.get(url, stream=True, timeout=self.timeout)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return {
                "success": True,
                "filepath": filepath,
                "size_bytes": os.path.getsize(filepath)
            }
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return {"success": False, "error": str(e)}
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close_driver()
