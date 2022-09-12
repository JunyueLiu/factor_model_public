import time

import pandas as pd
import requests
from bs4 import BeautifulSoup


def parse_table(soup):
    rows = soup.table.find_all('tr')[2:]
    data = []
    for r in rows:
        col = r.findAll('td')
        # https://stock.finance.sina.com.cn/stock/go.php/vReport_Show/kind/search/rptid/695192613128/index.phtml
        title_attrs = col[1].a.attrs
        report_url = 'https:' + title_attrs.get('href')
        title = title_attrs.get('title').split('：')[-1]
        pub_date = col[3].text
        ins = col[4].a.span.text
        analysts = col[5].span.text.replace('/', ',')
        data.append({
            'title': title,
            'url': report_url,
            'pub_date': pub_date,
            'institution': ins,
            'analyst_names': analysts
        })
    return data


def get_report(code: str, page):
    # http://stock.finance.sina.com.cn/stock/go.php/vReport_List/kind/search/index.phtml?symbol=300759&t1=all&p=6
    url = 'http://stock.finance.sina.com.cn/stock/go.php/vReport_List/kind/search/index.phtml' \
          '?symbol={}&t1=all&p={}' \
        .format(code, page)
    header = {'User-Agent':
                  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
              'Accept-Encoding': 'gzip, deflate',
              'Cache-Control': 'max-age=0',
              'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9'
              }
    while True:
        try:
            r = requests.get(url, headers=header)
            if r.status_code == 200:
                soup = BeautifulSoup(r.content, )
                # pd.read_html(StringIO(str(soup.table)))[0]
                if '没有找到相关内容' in soup.table.find_all('tr')[-1].td.text:
                    return None
                data = parse_table(soup)
                data = pd.DataFrame(data)
                data['code'] = code
                return data
            else:
                print(r.status_code)
                time.sleep(1)
        except Exception as e:
            print(e)
            break


if __name__ == '__main__':
    get_report('300759', 2)
