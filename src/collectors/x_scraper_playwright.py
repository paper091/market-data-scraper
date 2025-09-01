import asyncio
from playwright.async_api import async_playwright
from datetime import datetime

async def scrape_tweets(query="#sensex", max_tweets=50):
    url = f"https://x.com/search?q={query}&src=typed_query&f=live"
    results = []

    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, timeout=60000)

        while len(results) < max_tweets:
            tweets = await page.locator("article").all()
            for t in tweets[len(results):]:  # only new ones
                try:
                    username = await t.locator("div[dir='ltr']").nth(0).inner_text()
                    content = await t.locator("div[lang]").inner_text()
                    ts_attr = await t.locator("time").get_attribute("datetime")
                    ts = datetime.fromisoformat(ts_attr.replace("Z", "+00:00")) if ts_attr else None

                    # Engagement counts (likes, retweets, replies, quotes)
                    spans = await t.locator("div[role='group'] span").all_inner_texts()
                    counts = [int(s.replace(",", "")) if s.isdigit() else 0 for s in spans]

                    results.append({
                        "username": username,
                        "timestamp": ts.isoformat() if ts else None,
                        "content": content,
                        "reply_count": counts[0] if len(counts) > 0 else 0,
                        "retweet_count": counts[1] if len(counts) > 1 else 0,
                        "like_count": counts[2] if len(counts) > 2 else 0,
                        "quote_count": counts[3] if len(counts) > 3 else 0,
                    })
                except Exception:
                    continue

            await page.mouse.wheel(0, 2000)
            await asyncio.sleep(2)

            if len(results) >= max_tweets:
                break

        await browser.close()

    return results
