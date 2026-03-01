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
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


TOKEN_PATH = os.environ.get("KITE_TOKEN_PATH", "./secrets/kite_access_token.json")
DEBUG_DIR = os.environ.get("DEBUG_DIR", "./debug_login")
HEADLESS = os.environ.get("HEADLESS", "0").strip().lower() in {"1", "true", "yes"}

# These are set by the workflow to avoid version mismatch:
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


def first_visible(driver, candidates, timeout=12):
    end = time.time() + timeout
    last = None
    while time.time() < end:
        for by, sel in candidates:
            try:
                el = driver.find_element(by, sel)
                if el and el.is_displayed():
                    return el
            except Exception as e:
                last = e
        time.sleep(0.2)
    if last:
        raise last
    raise RuntimeError("Element not found")


def click_first(driver, candidates, timeout=12):
    el = first_visible(driver, candidates, timeout=timeout)
    el.click()
    return el


def type_first(driver, candidates, text, timeout=12, clear=True):
    el = first_visible(driver, candidates, timeout=timeout)
    if clear:
        try:
            el.clear()
        except Exception:
            pass
    el.send_keys(text)
    return el


def main():
    api_key = need_env("KITE_API_KEY")
    api_secret = need_env("KITE_API_SECRET")
    user_id = need_env("KITE_USER_ID")
    password = need_env("KITE_PASSWORD")

    kite_pin = os.environ.get("KITE_PIN", "").strip()
    totp_secret = os.environ.get("KITE_TOTP_SECRET", "").strip()
    if not kite_pin and not totp_secret:
        raise RuntimeError("Provide either KITE_PIN or KITE_TOTP_SECRET")

    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()

    chrome_opts = Options()
    if HEADLESS:
        chrome_opts.add_argument("--headless=new")
    chrome_opts.add_argument("--no-sandbox")
    chrome_opts.add_argument("--disable-dev-shm-usage")
    chrome_opts.add_argument("--window-size=1280,900")
    chrome_opts.add_argument("--disable-gpu")

    # Critical: force the Chrome binary installed by setup-chrome (avoids 145 vs 146 mismatch)
    if CHROME_BIN:
        chrome_opts.binary_location = CHROME_BIN

    # Critical: force the matching chromedriver installed by setup-chrome
    service = Service(executable_path=CHROMEDRIVER_BIN) if CHROMEDRIVER_BIN else Service()

    driver = webdriver.Chrome(service=service, options=chrome_opts)
    wait = WebDriverWait(driver, 45)

    try:
        driver.get(login_url)

        # Login screen
        type_first(driver, [(By.ID, "userid")], user_id)
        type_first(driver, [(By.ID, "password")], password)
        click_first(driver, [(By.CSS_SELECTOR, "button[type='submit']")])

        time.sleep(0.6)

        # 2FA PIN vs TOTP
        pin_mode = False
        try:
            first_visible(driver, [(By.ID, "pin")], timeout=8)
            pin_mode = True
        except Exception:
            pin_mode = False

        if pin_mode:
            if not kite_pin:
                raise RuntimeError("2FA is PIN but KITE_PIN not provided")
            type_first(driver, [(By.ID, "pin")], kite_pin)
            click_first(driver, [(By.CSS_SELECTOR, "button[type='submit']")])
        else:
            if not totp_secret:
                raise RuntimeError("2FA is TOTP but KITE_TOTP_SECRET not provided")
            otp = pyotp.TOTP(totp_secret).now()
            otp_box = first_visible(driver, [
                (By.CSS_SELECTOR, "input[placeholder*='OTP']"),
                (By.CSS_SELECTOR, "input[placeholder*='TOTP']"),
                (By.CSS_SELECTOR, "input[type='number']"),
                (By.CSS_SELECTOR, "input[type='text']"),
            ], timeout=12)
            otp_box.send_keys(otp)
            click_first(driver, [(By.CSS_SELECTOR, "button[type='submit']")])

        # Redirect with request_token
        wait.until(lambda d: "request_token=" in (d.current_url or ""))
        request_token = extract_request_token(driver.current_url)

        # request_token -> access_token (official flow)
        session = kite.generate_session(request_token, api_secret=api_secret)
        access_token = session["access_token"]

        mkdirp_for_file(TOKEN_PATH)
        payload = {
            "api_key": api_key,
            "generated_at": int(time.time()),
            "user_id": session.get("user_id"),
            "access_token": access_token,
        }
        with open(TOKEN_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"OK: token saved -> {TOKEN_PATH}")
        print(f"user_id={payload.get('user_id')} token_last4={access_token[-4:]}")

    except Exception:
        save_debug(driver, "failure")
        print("ERROR: Selenium login failed")
        traceback.print_exc()
        raise
    finally:
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
