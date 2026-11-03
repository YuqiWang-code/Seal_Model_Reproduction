import requests
from bs4 import BeautifulSoup
import re
import os

# 目标页面 URL
page_url = "https://max.book118.com/html/2023/0223/8024075045005040.shtm"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# 第一步：请求页面，解析出 PDF 地址
resp = requests.get(page_url, headers=headers)
resp.raise_for_status()
html = resp.text

# 方法1：寻找 .pdf 链接
pdf_urls = re.findall(r'https?://[^\s\'"]+\.pdf', html)
if not pdf_urls:
    # 方法2：寻找可能嵌入在 script 里的 base64 或转接链接
    soup = BeautifulSoup(html, "html.parser")
    # 下面这行可根据具体结构微调
    for script in soup.find_all("script"):
        txt = script.string or ""
        matches = re.findall(r'href="([^"]+\.pdf\?[^"]+)"', txt)
        pdf_urls.extend(matches)

if not pdf_urls:
    print("未找到 PDF 链接，请检查页面结构或网络请求。")
    exit(1)

# 去重
pdf_urls = list(dict.fromkeys(pdf_urls))
print("找到以下 PDF 链接：")
for i, u in enumerate(pdf_urls):
    print(f"{i+1}. {u}")

# 第二步：下载 PDF
for idx, pdf_url in enumerate(pdf_urls, 1):
    local_name = os.path.basename(pdf_url.split("?")[0])
    local_name = f"{idx:02d}_{local_name}"
    try:
        r = requests.get(pdf_url, headers=headers, stream=True)
        r.raise_for_status()
        with open(local_name, 'wb') as f:
            for chunk in r.iter_content(1024*1024):
                f.write(chunk)
        print(f"[√] 成功下载：{local_name}")
    except Exception as e:
        print(f"[×] 下载失败：{pdf_url}，原因：{e}")
