import pymongo
import pandas as pd

client = pymongo.MongoClient("mongodb://127.0.0.1/")
db = client["ecommercedb"]
collection = db["products"]

# Lấy tất cả sản phẩm có isActive là True và chỉ lấy id, name, và categoryId
query = {"isSelling": True}
projection = {"_id": 1, "name": 1}
products = collection.find(query, projection)

# Tạo DataFrame từ dữ liệu sản phẩm
product_data = list(products)
df = pd.DataFrame(product_data)

client.close()

# Lưu DataFrame vào tệp CSV
df.to_csv("data_set.csv", index=False)