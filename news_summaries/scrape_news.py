import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def get_card_data(card):    
    
    href = card.get("href", "").strip()
    title_el = card.select_one(".card__title-text")
    title = title_el.get_text(strip=True) if title_el else ""

    # Tag from data-tag on card__content
    content = card.select_one("div.card__content")
    tag = content.get("data-tag", "").strip() if content and content.has_attr("data-tag") else ""

    # Author from data-byline; normalize "By ..."
    byline_el = card.select_one("div.card__byline.mntl-card__byline")
    author_raw = byline_el.get("data-byline", "").strip() if byline_el and byline_el.has_attr("data-byline") else ""
    author = author_raw[3:]
    return href, title, tag, author

def get_news_data_from_card(href, session=None):
    sess = session or requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }
    resp = sess.get(href, headers=headers, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # Title
    title_el = soup.select_one("h1.article-heading")
    art_title = title_el.get_text(strip=True) if title_el else ""  # [web:45][web:53]

    # Author name + profile URL
    author_link = soup.select_one("div.mntl-bylines__group a.mntl-attribution__item-name")
    author_name = author_link.get_text(strip=True) if author_link else ""  # [web:97][web:45]
    author_url = urljoin(href, author_link.get("href", "")) if author_link else ""  # [web:45][web:53]

    # Published date and time
    date_el = soup.select_one("div.mntl-attribution__item-date")
    published_date = date_el.get_text(strip=True) if date_el else ""  # [web:97][web:45]
    time_el = soup.select_one("div.timestamp > span.timestamp")
    published_time = time_el.get_text(strip=True) if time_el else ""  # [web:97][web:45]

    # Primary image and caption
    img_el = soup.select_one("figure.article-primary-image img.primary-image__image, figure.article-primary-image img")
    primary_image_url = ""
    if img_el:
        primary_image_url = img_el.get("src") or img_el.get("data-src") or ""  # [web:53][web:45]
    cap_el = soup.select_one("figure.article-primary-image figcaption .figure-article-caption-text")
    primary_caption = cap_el.get_text(strip=True) if cap_el else ""  # [web:53][web:45]

    # Key Takeaways list
    key_takeaways = []
    kt_body = soup.select_one(".finance-sc-block-callout--whatyouneedtoknow .mntl-sc-block-universal-callout__body")
    if kt_body:
        for li in kt_body.select("li"):
            key_takeaways.append(li.get_text(" ", strip=True))  # [web:53][web:45]

    # Content sections in reading order: headings and paragraphs
    sections = []
    content_root = soup.select_one("#article-body_1-0, .article-body")
    if content_root:
        for node in content_root.find_all(recursive=False):
            for child in getattr(node, "descendants", []):
                if getattr(child, "name", None) == "h2" and "finance-sc-block-heading" in " ".join(child.get("class", [])):
                    txt_el = child.select_one(".mntl-sc-block-heading__text")
                    if txt_el:
                        sections.append({"type": "heading", "text": txt_el.get_text(" ", strip=True)})  # [web:53][web:45]
                elif getattr(child, "name", None) == "p" and "finance-sc-block-html" in " ".join(child.get("class", [])):
                    sections.append({"type": "paragraph", "text": child.get_text(" ", strip=True)})  # [web:53][web:45]

    # Citations from "Article Sources"
    citations = []
    for li in soup.select(".mntl-article-sources__citation-sources-1 .mntl-sources__content li.mntl-sources__source"):
        cid = li.get("id", "")
        a = li.select_one("a")
        c_title = a.get_text(" ", strip=True) if a else li.get_text(" ", strip=True)
        c_url = a.get("href", "") if a else ""
        citations.append({"id": cid, "title": c_title, "url": c_url})  # [web:53][web:45]

    # Related stories
    related = []
    for a in soup.select("#midcirc__card-list_1-0 a.midcirc-card"):
        rt = a.select_one(".midcirc-card__title")
        r_title = rt.get_text(strip=True) if rt else ""
        r_href = urljoin(href, a.get("href", ""))
        r_img_el = a.select_one("img")
        r_img = ""
        if r_img_el:
            r_img = r_img_el.get("src") or r_img_el.get("data-src") or ""
        related.append({"title": r_title, "url": r_href, "image": r_img})  # [web:53][web:45]

    # Breadcrumbs
    breadcrumbs = []
    for b in soup.select("ul.mntl-universal-breadcrumbs li a.mntl-breadcrumbs__link"):
        breadcrumbs.append({"text": b.get_text(strip=True), "url": urljoin(href, b.get("href", ""))})  # [web:53][web:45]

    return {
        "url": href,
        "title": art_title,
        "author_name": author_name,
        "author_url": author_url,
        "published_date": published_date,
        "published_time": published_time,
        "primary_image_url": primary_image_url,
        "primary_caption": primary_caption,
        "key_takeaways": key_takeaways,
        "sections": sections,
        "citations": citations,
        "related": related,
        "breadcrumbs": breadcrumbs,
    }
    

URL = "https://www.investopedia.com/news-4427706"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

resp = requests.get(URL, headers=headers, timeout=30)
resp.raise_for_status()
soup = BeautifulSoup(resp.text, "lxml")
cards = soup.select("a.mntl-card-list-items.mntl-universal-card.mntl-document-card.mntl-card")

session = requests.Session()
rows = []
for card in cards:
    href, title, tag, author = get_card_data(card)
    detail = get_news_data_from_card(href, session=session)
    detail.update({"card_title": title, "card_tag": tag, "card_author": author})
    rows.append(detail) 

print(rows[0])