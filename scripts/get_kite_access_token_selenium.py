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

# Provided by workflow to avoid Chrome/Driver mismatch:
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

    opts = Options()
    if HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1280,900")
    opts.add_argument("--disable-gpu")

    # Force correct Chrome binary (avoids /usr/bin/google-chrome 145.x)
    if CHROME_BIN:
        opts.binary_location = CHROME_BIN  # set chrome binary location :contentReference[oaicite:3]{index=3}

    # Force correct chromedriver executable (Service class) :contentReference[oaicite:4]{index=4}
    service = Service(executable_path=CHROMEDRIVER_BIN) if CHROMEDRIVER_BIN else Service()

    driver = webdriver.Chrome(service=service, options=opts)
    wait = WebDriverWait(driver, 45)

    try:
        driver.get(login_url)

        wait.until(EC.visibility_of_element_located((By.ID, "userid"))).send_keys(user_id)
        wait.until(EC.visibility_of_element_located((By.ID, "password"))).send_keys(password)
        wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']"))).click()

        time.sleep(0.6)

        # PIN vs TOTP
        pin_mode = False
        try:
            WebDriverWait(driver, 8).until(EC.visibility_of_element_located((By.ID, "pin")))
            pin_mode = True
        except Exception:
            pin_mode = False

        if pin_mode:
            if not kite_pin:
                raise RuntimeError("PIN required but KITE_PIN not provided")
            wait.until(EC.visibility_of_element_located((By.ID, "pin"))).send_keys(kite_pin)
            wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']"))).click()
        else:
            if not totp_secret:
                raise RuntimeError("TOTP required but KITE_TOTP_SECRET not provided")
            otp = pyotp.TOTP(totp_secret).now()

            otp_box = None
            for sel in [
                (By.CSS_SELECTOR, "input[placeholder*='OTP']"),
                (By.CSS_SELECTOR, "input[placeholder*='TOTP']"),
                (By.CSS_SELECTOR, "input[type='number']"),
                (By.CSS_SELECTOR, "input[type='text']"),
            ]:
                try:
                    otp_box = WebDriverWait(driver, 10).until(EC.visibility_of_element_located(sel))
                    break
                except Exception:
                    pass
            if otp_box is None:
                raise RuntimeError("Could not find OTP input box")
            otp_box.send_keys(otp)
            wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']"))).click()

        wait.until(lambda d: "request_token=" in (d.current_url or ""))
        request_token = extract_request_token(driver.current_url)

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
        save_debug(driver, "failure")
        traceback.print_exc()
        raise
    finally:
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
