#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import traceback
from urllib.parse import urlparse, parse_qs

import pyotp
from kiteconnect import KiteConnect

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


TOKEN_PATH = os.environ.get("KITE_TOKEN_PATH", "./secrets/kite_access_token.json")
DEBUG_DIR = os.environ.get("DEBUG_DIR", "./debug_login")

# run headed under Xvfb in Actions
HEADLESS = os.environ.get("HEADLESS", "0").strip().lower() in {"1", "true", "yes"}

# Provided by workflow to avoid Chrome/Driver mismatch
CHROME_BIN = os.environ.get("CHROME_BIN", "").strip()
CHROMEDRIVER_BIN = os.environ.get("CHROMEDRIVER_BIN", "").strip()


def need_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v.strip()


def mkdirp(path: str):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def mkdirp_for_file(path: str):
    mkdirp(os.path.dirname(os.path.abspath(path)))


def extract_request_token(url: str) -> str:
    q = parse_qs(urlparse(url).query)
    rt = (q.get("request_token") or [None])[0]
    if not rt:
        raise RuntimeError(f"request_token not found in URL: {url}")
    return rt


def save_debug(driver, tag: str):
    mkdirp(DEBUG_DIR)
    try:
        driver.save_screenshot(os.path.join(DEBUG_DIR, f"{tag}.png"))
    except Exception:
        pass
    try:
        with open(os.path.join(DEBUG_DIR, f"{tag}.html"), "w", encoding="utf-8") as f:
            f.write(driver.page_source or "")
    except Exception:
        pass
    try:
        with open(os.path.join(DEBUG_DIR, f"{tag}.url.txt"), "w", encoding="utf-8") as f:
            f.write(driver.current_url or "")
    except Exception:
        pass


def js_click(driver, el):
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
    driver.execute_script("arguments[0].click();", el)


def first_present(driver, locators, timeout=12):
    end = time.time() + timeout
    last = None
    while time.time() < end:
        for by, sel in locators:
            try:
                el = driver.find_element(by, sel)
                if el:
                    return el
            except Exception as e:
                last = e
        time.sleep(0.2)
    if last:
        raise last
    raise RuntimeError("element not found")


def first_visible(driver, locators, timeout=12):
    end = time.time() + timeout
    last = None
    while time.time() < end:
        for by, sel in locators:
            try:
                el = driver.find_element(by, sel)
                if el and el.is_displayed():
                    return el
            except Exception as e:
                last = e
        time.sleep(0.2)
    if last:
        raise last
    raise RuntimeError("visible element not found")


def try_accept_overlays(driver):
    # Best-effort: if any obvious consent modal exists, click it.
    candidates = [
        (By.XPATH, "//button[contains(.,'Accept') or contains(.,'I Agree') or contains(.,'OK')]"),
        (By.CSS_SELECTOR, "button#accept"),
        (By.CSS_SELECTOR, "button[aria-label*='Accept']"),
    ]
    for by, sel in candidates:
        try:
            el = driver.find_element(by, sel)
            if el and el.is_displayed():
                js_click(driver, el)
                time.sleep(0.3)
                return True
        except Exception:
            pass
    return False


def click_login_submit(driver, wait: WebDriverWait):
    """
    Robust click:
    - multiple selectors
    - wait for presence then JS click fallback
    """
    submit_locators = [
        (By.CSS_SELECTOR, "button[type='submit']"),
        (By.CSS_SELECTOR, "button.button-orange"),
        (By.XPATH, "//button[contains(.,'Login') or contains(.,'Continue') or @type='submit']"),
    ]

    # Try overlays first
    try_accept_overlays(driver)

    # Try normal clickable wait on each selector
    last_exc = None
    for loc in submit_locators:
        try:
            btn = wait.until(EC.presence_of_element_located(loc))
            # If selenium thinks it's clickable, use normal click
            try:
                wait.until(EC.element_to_be_clickable(loc)).click()
                return
            except Exception:
                # fallback to JS click
                js_click(driver, btn)
                return
        except Exception as e:
            last_exc = e

    # Last resort: press Enter on password field (many forms submit)
    try:
        pw = driver.find_element(By.ID, "password")
        pw.send_keys(Keys.ENTER)
        return
    except Exception:
        pass

    raise last_exc if last_exc else RuntimeError("Could not click submit")


def main():
    api_key = need_env("KITE_API_KEY")
    api_secret = need_env("KITE_API_SECRET")
    user_id = need_env("KITE_USER_ID")
    password = need_env("KITE_PASSWORD")

    # Prefer TOTP (your current runs are in TOTP mode)
    totp_secret = os.environ.get("KITE_TOTP_SECRET", "").strip()
    kite_pin = os.environ.get("KITE_PIN", "").strip()

    if not kite_pin and not totp_secret:
        raise RuntimeError("Provide either KITE_PIN or KITE_TOTP_SECRET")

    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()

    opts = Options()
    if HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1280,900")
    opts.add_argument("--disable-gpu")

    # Force the Chrome binary installed by setup-chrome (avoid /usr/bin/google-chrome mismatch)
    if CHROME_BIN:
        opts.binary_location = CHROME_BIN

    service = Service(executable_path=CHROMEDRIVER_BIN) if CHROMEDRIVER_BIN else Service()
    driver = webdriver.Chrome(service=service, options=opts)
    wait = WebDriverWait(driver, 45)

    try:
        driver.get(login_url)
        save_debug(driver, "00_opened_login_url")

        # Wait for userid/password fields (Kite login uses these inputs)
        uid = wait.until(EC.visibility_of_element_located((By.ID, "userid")))
        pw = wait.until(EC.visibility_of_element_located((By.ID, "password")))

        uid.clear()
        uid.send_keys(user_id)
        pw.clear()
        pw.send_keys(password)
        save_debug(driver, "01_filled_user_pass")

        # Click Login (robust)
        try:
            click_login_submit(driver, wait)
        except TimeoutException:
            save_debug(driver, "02_submit_timeout")
            raise
        save_debug(driver, "03_after_submit")

        time.sleep(0.8)

        # 2FA: Kite can show PIN input OR TOTP/app-code input.
        # First try PIN:
        try:
            pin_el = WebDriverWait(driver, 6).until(EC.visibility_of_element_located((By.ID, "pin")))
            if not kite_pin:
                raise RuntimeError("PIN required but KITE_PIN not provided")
            pin_el.clear()
            pin_el.send_keys(kite_pin)
            click_login_submit(driver, wait)
            save_debug(driver, "04_pin_submitted")
        except Exception:
            # Otherwise TOTP/app-code:
            if not totp_secret:
                save_debug(driver, "04_totp_missing_secret")
                raise RuntimeError("TOTP required but KITE_TOTP_SECRET not provided")
            otp = pyotp.TOTP(totp_secret).now()

            otp_box = None
            for loc in [
                (By.CSS_SELECTOR, "input[placeholder*='App']"),
                (By.CSS_SELECTOR, "input[placeholder*='code']"),
                (By.CSS_SELECTOR, "input[placeholder*='OTP']"),
                (By.CSS_SELECTOR, "input[type='number']"),
                (By.CSS_SELECTOR, "input[type='text']"),
            ]:
                try:
                    otp_box = WebDriverWait(driver, 10).until(EC.visibility_of_element_located(loc))
                    break
                except Exception:
                    pass

            if otp_box is None:
                save_debug(driver, "04_totp_input_not_found")
                raise RuntimeError("Could not find TOTP/App-code input")

            otp_box.clear()
            otp_box.send_keys(otp)
            click_login_submit(driver, wait)
            save_debug(driver, "05_totp_submitted")

        # Redirect with request_token
        wait.until(lambda d: "request_token=" in (d.current_url or ""))
        save_debug(driver, "06_got_request_token_url")

        request_token = extract_request_token(driver.current_url)

        # Exchange request_token -> access_token (official flow)
        session = kite.generate_session(request_token, api_secret=api_secret)
        access_token = session["access_token"]

        mkdirp_for_file(TOKEN_PATH)
        with open(TOKEN_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "api_key": api_key,
                    "generated_at": int(time.time()),
                    "user_id": session.get("user_id"),
                    "access_token": access_token,
                },
                f,
                indent=2,
            )

        print(f"OK: token saved -> {TOKEN_PATH}")
        print(f"user_id={session.get('user_id')} token_last4={access_token[-4:]}")

    except Exception:
        try:
            save_debug(driver, "zz_failure")
        except Exception:
            pass
        traceback.print_exc()
        raise
    finally:
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
