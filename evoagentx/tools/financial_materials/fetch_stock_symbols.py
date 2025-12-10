import requests
import json
import time

API_KEY = "aae67179e4a95f29a213301b8aa69af3"
BASE_URL = "https://financialdata.net/api/v1/stock-symbols"

all_symbols = []
offset = 0
limit = 500

print("开始获取 Stock Symbols...")

while True:
    url = f"{BASE_URL}?offset={offset}&format=json&key={API_KEY}"
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        if not data or len(data) == 0:
            print(f"offset={offset} 没有更多数据")
            break
        
        all_symbols.extend(data)
        print(f"offset={offset}: +{len(data)} 条, 累计: {len(all_symbols)}")
        
        if len(data) < limit:
            print("数据不足 500 条，已获取全部")
            break
        
        offset += limit
        time.sleep(0.3)
        
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        break

output_file = "stock_symbols.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_symbols, f, ensure_ascii=False, indent=2)

print(f"\n完成！共获取 {len(all_symbols)} 条 Stock Symbols")
print(f"已保存到: {output_file}")

