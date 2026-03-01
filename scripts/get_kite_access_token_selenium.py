#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import traceback
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import pyotp
from kiteconnect import KiteConnect

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# Output token JSON here
TOKEN_PATH = os.environ.get("KITE_TOKEN_PATH", "./secrets/kite_access_token.json")
DEBUG_DIR = os.environ.get("DEBUG_DIR", "./debug_login")

# Headed-by-default (use Xvfb in Actions)
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


def redact_url(url: str) -> str:
    """Redact request_token in logged URLs."""
    try:
        u = urlparse(url)
        q = parse_qs(u.query)
        if "request_token" in q:
            q["request_token"] = ["REDACTED"]
        new_query = urlencode({k: v[0] for k, v in q.items()}, doseq=False)
        return urlunparse((u.scheme, u.netloc, u.path, u.params, new_query, u.fragment))
    except Exception:
        return url


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
            f.write(redact_url(driver.current_url or ""))
    except Exception:
        pass


def js_click(driver, el):
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
    driver.execute_script("arguments[0].click();", el)


def try_click_submit(driver, wait: WebDriverWait):
    """Robust submit click: multiple selectors + JS click + Enter fallback."""
    locators = [
        (By.CSS_SELECTOR, "button[type='submit']"),
        (By.CSS_SELECTOR, "button.button-orange"),
        (By.XPATH, "//button[contains(.,'Login') or contains(.,'Continue') or @type='submit']"),
    ]

    last = None
    for loc in locators:
        try:
            btn = wait.until(EC.presence_of_element_located(loc))
            try:
                wait.until(EC.element_to_be_clickable(loc)).click()
            except Exception:
                js_click(driver, btn)
            return True
        except Exception as e:
            last = e

    # last resort: press Enter on active element
    try:
        driver.switch_to.active_element.send_keys(Keys.ENTER)
        return True
    except Exception:
        if last:
            raise last
        raise RuntimeError("Could not click submit")


def maybe_request_token(driver) -> str | None:
    url = driver.current_url or ""
    return extract_request_token(url) if "request_token=" in url else None


def wait_for_request_token_or_2fa(driver, wait: WebDriverWait, timeout_s: int = 90) -> str:
    """
    After clicking login, the flow can be:
      A) Direct redirect to callback with request_token (even if callback is localhost and refuses)
      B) Show PIN page
      C) Show TOTP/App-code page
    We loop until we obtain request_token, otherwise we try to complete 2FA if fields appear.
    """
    kite_pin = os.environ.get("KITE_PIN", "").strip()
    totp_secret = os.environ.get("KITE_TOTP_SECRET", "").strip()

    end = time.time() + timeout_s
    last_stage = None

    while time.time() < end:
        # 1) If request_token is already in URL, we're done (even if page doesn't load)
        rt = maybe_request_token(driver)
        if rt:
            return rt

        # 2) PIN page?
        try:
            pin_el = driver.find_element(By.ID, "pin")
            if pin_el.is_displayed():
                last_stage = "pin"
                if not kite_pin:
                    raise RuntimeError("PIN required but KITE_PIN not provided")
                pin_el.clear()
                pin_el.send_keys(kite_pin)
                try_click_submit(driver, wait)
                time.sleep(0.6)
                continue
        except Exception:
            pass

        # 3) TOTP/App-code page?
        # (Kite uses “App code”/TOTP in many setups; input ids can vary, so use broad locators.)
        otp_locators = [
            (By.CSS_SELECTOR, "input[placeholder*='App']"),
            (By.CSS_SELECTOR, "input[placeholder*='code']"),
            (By.CSS_SELECTOR, "input[placeholder*='OTP']"),
            (By.CSS_SELECTOR, "input[type='number']"),
        ]
        otp_el = None
        for by, sel in otp_locators:
            try:
                el = driver.find_element(by, sel)
                if el and el.is_displayed():
                    otp_el = el
                    break
            except Exception:
                pass

        if otp_el is not None:
            last_stage = "totp"
            if not totp_secret:
                raise RuntimeError("TOTP required but KITE_TOTP_SECRET not provided")
            otp = pyotp.TOTP(totp_secret).now()
            try:
                otp_el.clear()
            except Exception:
                pass
            otp_el.send_keys(otp)
            try_click_submit(driver, wait)
            time.sleep(0.6)
            continue

        # 4) Nothing obvious yet; keep waiting
        if last_stage != "waiting":
            last_stage = "waiting"
        time.sleep(0.4)

    # final check
    rt = maybe_request_token(driver)
    if rt:
        return rt

    raise RuntimeError("Timed out waiting for request_token or 2FA inputs")


def main():
    api_key = need_env("KITE_API_KEY")
    api_secret = need_env("KITE_API_SECRET")
    user_id = need_env("KITE_USER_ID")
    password = need_env("KITE_PASSWORD")

    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()

    opts = Options()
    if HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1280,900")
    opts.add_argument("--disable-gpu")

    # Force Chrome binary & matching driver from setup-chrome
    if CHROME_BIN:
        opts.binary_location = CHROME_BIN
    service = Service(executable_path=CHROMEDRIVER_BIN) if CHROMEDRIVER_BIN else Service()

    driver = webdriver.Chrome(service=service, options=opts)
    wait = WebDriverWait(driver, 45)

    try:
        driver.get(login_url)
        save_debug(driver, "00_opened_login_url")

        # Login fields
        uid = wait.until(EC.visibility_of_element_located((By.ID, "userid")))
        pw = wait.until(EC.visibility_of_element_located((By.ID, "password")))
        uid.clear()
        uid.send_keys(user_id)
        pw.clear()
        pw.send_keys(password)
        save_debug(driver, "01_filled_user_pass")

        # Click Login
        try_click_submit(driver, wait)
        save_debug(driver, "02_clicked_login")

        # Wait for request_token (or handle 2FA if it appears)
        request_token = wait_for_request_token_or_2fa(driver, wait, timeout_s=90)
        save_debug(driver, "03_got_request_token_url")

        # Exchange request_token -> access_token (official flow)
        # request_token is returned to the configured redirect URL as a query param. :contentReference[oaicite:1]{index=1}
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
        save_debug(driver, "zz_failure")
        traceback.print_exc()
        raise
    finally:
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
