from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# 1. Inisialisasi SparkSession 
spark = SparkSession.builder.appName("Data Analysis").config("spark.executor.memory", "4g").config("spark.driver.memory", "2g").getOrCreate()

# 2. Load data
df = spark.read.csv("data/sales.csv", header=True, inferSchema=True)

# 3. Data transformation
result = df.select("product", "sales", "region").filter(col("sales") > 1000).groupBy("region").agg(sum("sales").alias("total_sales")).orderBy(col("total_sales").desc())

# 4. Action - trigger computation 
result.show()
result.coalesce(1).write.mode("overwrite").csv("output/regional_sales")

# 5. Cleanup 
spark.stop()

from pyspark import SparkContext, SparkConf

# Inisialisasi SparkContext
conf = SparkConf().setAppName("RDD Operations") 
sc = SparkContext(conf=conf)

# Membuat RDD dari list
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
rdd = sc.parallelize(numbers)

# Map: kuadratkan setiap angka 
squared_rdd = rdd.map(lambda x: x ** 2) 
print("Original:", rdd.collect()) 
print("Squared:", squared_rdd.collect())

# Map dengan fungsi lebih kompleks 
def categorize_number(n):
 if n % 2 == 0:
  return f"Even: {n}" 
 else:
  return f"Odd: {n}"

categorized_rdd = rdd.map(categorize_number) 
print("Categorized:", categorized_rdd.collect())

# Filter: ambil hanya angka genap
even_rdd = rdd.filter(lambda x: x % 2 == 0) 
print("Even numbers:", even_rdd.collect())

# Filter dengan multiple conditions
filtered_rdd = rdd.filter(lambda x: x > 3 and x < 8) 
print("Numbers between 3 and 8:", filtered_rdd.collect())

# Filter dengan string operations
words = ["hello", "world", "spark", "python", "data"] 
words_rdd = sc.parallelize(words)
long_words = words_rdd.filter(lambda word: len(word) > 4) 
print("Long words:", long_words.collect())

# Reduce: jumlahkan semua angka
total_sum = rdd.reduce(lambda a, b: a + b) 
print("Sum:", total_sum)

# Reduce: temukan nilai maksimum
max_value = rdd.reduce(lambda a, b: a if a > b else b) 
print("Maximum:", max_value)

# Reduce untuk string concatenation
words_rdd = sc.parallelize(["Hello", "World", "From", "Spark"]) 
concatenated = words_rdd.reduce(lambda a, b: a + " " + b) 
print("Concatenated:", concatenated)

# Simulasi data log web server
log_data = [
  "192.168.1.1 - [10/Oct/2023:13:55:36] GET /home.html 200 2326",
  "192.168.1.2 - [10/Oct/2023:13:55:37] GET /about.html 200 1534",
  "192.168.1.1 - [10/Oct/2023:13:55:38] POST /login 401 0",
  "192.168.1.3 - [10/Oct/2023:13:55:39] GET /products.html 200 3421",
  "192.168.1.2 - [10/Oct/2023:13:55:40] GET /contact.html 404 0",
  "192.168.1.4 - [10/Oct/2023:13:55:41] GET /home.html 200 2326",
  "192.168.1.1 - [10/Oct/2023:13:55:42] GET /admin 403 0",
  "192.168.1.5 - [10/Oct/2023:13:55:43] GET /api/data 500 0"
]

# Membuat RDD dari data log
log_rdd = sc.parallelize(log_data)

# Fungsi untuk parsing log entry dengan regex
import re

def parse_log_line(line):
  try:
    # Pattern: IP - [timestamp] METHOD URL STATUS SIZE
    pattern = r'^(\S+)\s+-\s+\[([^\]]+)\]\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)'
    match = re.match(pattern, line)
    
    if match:
      ip = match.group(1)
      timestamp = match.group(2)
      method = match.group(3)
      url = match.group(4)
      status_code = int(match.group(5))
      size = int(match.group(6))
      return (ip, timestamp, method, url, status_code, size)
    else:
      print(f"Failed to parse: {line}")
      return None
  except Exception as e:
    print(f"Error parsing line: {line}, Error: {e}")
    return None

# Parse semua log entries dan filter yang berhasil
parsed_logs = log_rdd.map(parse_log_line).filter(lambda x: x is not None)

# Cache hasil parsing untuk efisiensi
parsed_logs.cache()

# Analisis 1: Hitung request per IP address
ip_counts = (
  parsed_logs.map(lambda log: (log[0], 1))
    .reduceByKey(lambda a, b: a + b)
    .sortBy(lambda x: x[1], ascending=False)
)

print("Requests per IP Address:")
for ip, count in ip_counts.collect():
  print(f"{ip}: {count} requests")

# Analisis 2: Hitung error rate (status code >= 400)
total_requests = parsed_logs.count()
error_requests = parsed_logs.filter(lambda log: log[4] >= 400).count()
error_rate = (error_requests / total_requests) * 100 if total_requests > 0 else 0

print(f"\nTotal requests: {total_requests}")
print(f"Error requests: {error_requests}")
print(f"Error rate: {error_rate:.2f}%")

# Analisis 3: Top URLs berdasarkan traffic
url_counts = (
  parsed_logs.map(lambda log: (log[3], 1))
    .reduceByKey(lambda a, b: a + b)
    .sortBy(lambda x: x[1], ascending=False)
)

print("\nTop URLs:")
for url, count in url_counts.take(5):
  print(f"{url}: {count} requests")

# Analisis 4: Total bandwidth per status code
bandwidth_by_status = (
  parsed_logs.map(lambda log: (log[4], log[5]))
    .reduceByKey(lambda a, b: a + b)
    .sortBy(lambda x: x[0])
)

print("\nBandwidth by Status Code:")
for status, bandwidth in bandwidth_by_status.collect():
  print(f"Status {status}: {bandwidth} bytes")

import time

# Membuat RDD besar untuk demonstrasi
large_data = sc.parallelize(range(10000000))

print("=== Demonstrasi Lazy Evaluation ===")

# Tahap 1: Definisi transformations (lazy)
print("Mendefinisikan transformations...")
start_time = time.time()

filtered_data = large_data.filter(lambda x: x % 2 == 0)
mapped_data = filtered_data.map(lambda x: x * 2)
final_data = mapped_data.filter(lambda x: x > 1000)

transformation_time = time.time() - start_time
print(f"Waktu definisi transformations: {transformation_time:.4f} detik")
print("Tidak ada komputasi yang dilakukan pada tahap ini!")

# Tahap 2: Eksekusi action (eager)
print("\nMenjalankan action...")
start_time = time.time()

result_count = final_data.count() # Action yang memicu eksekusi
execution_time = time.time() - start_time

print(f"Waktu eksekusi actual: {execution_time:.4f} detik")
print(f"Hasil count: {result_count}")

# Tahap 3: Multiple actions pada RDD yang sama
print("\nMenjalankan action kedua...")
start_time = time.time()

first_elements = final_data.take(5) # Action kedua
second_action_time = time.time() - start_time

print(f"Waktu action kedua: {second_action_time:.4f} detik")
print(f"Elemen pertama: {first_elements}")

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *

# Inisialisasi SparkSession
spark = SparkSession.builder.appName("DataFrame Operations").config("spark.executor.memory", "4g").getOrCreate()

# 1. Basic CSV reading dengan schema inference
df_sales = spark.read.csv(
  "data/Sale Report.csv",
  header=True,
  inferSchema=True
)

print("=== Basic CSV Reading ===")
df_sales.printSchema()
print(f"Jumlah rows: {df_sales.count()}")
print(f"Jumlah columns: {len(df_sales.columns)}")

# 2. CSV reading dengan explicit schema (recommended untuk production)
sales_schema = StructType([
  StructField("index", IntegerType(), True),
  StructField("SKU_Code", StringType(), True),
  StructField("Design_No", StringType(), True),
  StructField("Stock", DoubleType(), True),
  StructField("Category", StringType(), True),
  StructField("Size", StringType(), True),
  StructField("Color", StringType(), True)
])

df_sales_typed = spark.read.csv("data/Sale Report.csv",header=True,schema=sales_schema)

# 3. Advanced CSV options untuk handling edge cases
df_amazon = spark.read.option("header", "true").option("inferSchema", "true").option("multiline", "true").option("escape", '"').csv("data/Amazon Sale Report.csv")

print("\n=== Amazon Sales Data ===")
df_amazon.printSchema()
df_amazon.show(5, truncate=False)

# Simulasi nested JSON structure
json_data = [
  {
    "product_id": "P001",
    "details": {
      "name": "Laptop Gaming",
      "category": "Electronics",
      "specs": {"ram": "16GB", "storage": "512GB SSD"}
    },
    "sales": [
      {"date": "2023-01-01", "quantity": 5, "amount": 75000000},
      {"date": "2023-01-02", "quantity": 3, "amount": 45000000}
    ]
  },
  {
    "product_id": "P002",
    "details": {
      "name": "Smartphone",
      "category": "Electronics",
      "specs": {"ram": "8GB", "storage": "128GB"}
    },
    "sales": [
      {"date": "2023-01-01", "quantity": 10, "amount": 50000000}
    ]
  }
]

# Create DataFrame from JSON-like data
df_json_nested = spark.createDataFrame(json_data)

print("=== Nested JSON DataFrame ===")
df_json_nested.printSchema()
df_json_nested.show(truncate=False)

# Flatten the 'sales' array and calculate total sales
df_json_processed = df_json_nested.withColumn("sale", explode(col("sales"))) \
  .select(
      col("product_id"),
      col("details.name").alias("product_name"),
      col("details.category").alias("category"),
      col("sale.amount").alias("sale_amount")
  ) \
  .groupBy("product_id", "product_name", "category") \
  .agg(sum("sale_amount").alias("total_sales"))

print("\n=== Processed DataFrame with Total Sales ===")
df_json_processed.show()

# Load data untuk demonstrasi
df_sales = spark.read.csv("data/Sale Report.csv", header=True, inferSchema=True)

# 1. Basic column selection
selected_cols = df_sales.select("SKU Code", "Category", "Size", "Stock")
print("=== Basic Column Selection ===")
selected_cols.show(10)

# 2. Column selection dengan renaming
renamed_cols = df_sales.select(
  col("SKU Code").alias("product_sku"),
  col("Category").alias("product_category"),
  col("Stock").alias("inventory_count"),
  col("Color").alias("product_color")
)
print("\n=== Renamed Columns ===")
renamed_cols.show(5)

# 3. Select dengan computed columns
computed_cols = df_sales.select("SKU Code", "Category","Stock", (col("Stock") * 1000).alias("stock_value_estimate"), when(col("Stock") > 5, "High Stock").when(col("Stock") > 2, "Medium Stock").otherwise("Low Stock").alias("stock_level"))
print("\n=== Computed Columns ===")
computed_cols.show(10)

# 1. Simple filtering
high_stock = df_sales.filter(col("Stock") > 5)
print("=== High Stock Products ===")
print(f"Total products with high stock: {high_stock.count()}")
high_stock.select("SKU Code", "Category", "Stock").show(10)

# 2. Multiple conditions dengan AND
filtered_products = df_sales.filter(
  (col("Stock") > 3) & (col("Category").contains("LEGGINGS"))
)
print("\n=== Filtered Products (Stock > 3 AND Category contains LEGGINGS) ===")
filtered_products.show(10)

# 3. Complex filtering dengan OR dan string operations
complex_filter = df_sales.filter((col("Color") == "Red") | (col("Color") == "Blue") | (col("Size").isin(["L", "XL"])))
print("\n=== Complex Filter Results ===")
complex_filter.groupBy("Color", "Size").count().orderBy("count", ascending=False).show()

# 1. Basic groupBy dengan single aggregation
category_analysis = df_sales.groupBy("Category").agg(count("*").alias("product_count"),sum("Stock").alias("total_stock"), avg("Stock").alias("avg_stock")
  ).orderBy(col("total_stock").desc())

print("=== Category Analysis ===")
category_analysis.show()

# 2. Multiple groupBy columns
size_color_analysis = df_sales.groupBy("Size", "Color").agg(count("*").alias("sku_count"),sum("Stock").alias("total_inventory")
  ).filter(col("sku_count") > 2).orderBy(col("total_inventory").desc())

print("\n=== Size-Color Combination Analysis ===")
size_color_analysis.show()

# 3. Advanced aggregations dengan custom functions
from pyspark.sql.functions import stddev, variance, collect_list

detailed_analysis = df_sales.groupBy("Category").agg(
count("*").alias("product_count"), sum("Stock").alias("total_stock"), avg("Stock").alias("avg_stock"),stddev("Stock").alias("stock_stddev"), min("Stock").alias("min_stock"), max("Stock").alias("max_stock"),collect_list("Color").alias("available_colors"))

print("\n=== Detailed Category Analysis ===")
detailed_analysis.show(truncate=False)

# Register DataFrame sebagai temporary view
df_sales.createOrReplaceTempView("sales_inventory")

# Basic SQL queries
inventory_summary = spark.sql(""" SELECT Category, COUNT(*) AS product_count, SUM(Stock) AS total_stock, AVG(Stock) AS avg_stock FROM sales_inventory GROUP BY Category ORDER BY total_stock DESC
""")

print("=== Inventory Summary ===")
inventory_summary.show()

# Complex SQL dengan window functions
top_categories = spark.sql("""
WITH category_stats AS (SELECT Category, COUNT(*) AS product_count, SUM(Stock) AS total_stock, ROW_NUMBER() OVER (ORDER BY SUM(Stock) DESC) AS rank FROM sales_inventory GROUP BY Category)
SELECT Category, product_count, total_stock
FROM category_stats WHERE rank <= 5
""")

print("=== Top 5 Categories ===")
top_categories.show()

from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# Inisialisasi Spark
spark = SparkSession.builder.appName("E-commerce Analysis").config("spark.executor.memory", "2g").getOrCreate()

# Load datasets
df_sales = spark.read.csv("data/Sale Report.csv", header=True, inferSchema=True)
df_amazon = spark.read.csv("data/Amazon Sale Report.csv", header=True, inferSchema=True)
print(f"Inventory data: {df_sales.count():,} rows, {len(df_sales.columns)} columns")
print(f"Amazon sales: {df_amazon.count():,} rows, {len(df_amazon.columns)} columns")

# Data quality check
print("Missing values in Amazon sales:")
df_amazon.select([count(when(col(c).isNull(), c)).alias(c) for c in ["Category", "Amount", "Status"]]).show()

# Basic statistics
df_amazon.select("Amount", "Qty").describe().show()

# 1. Clean data Amazon sales
df_amazon_clean = df_amazon.filter(col("Amount").isNotNull() & (col("Amount") > 0) & col("Status").isNotNull()).withColumn("is_delivered",when(col("Status").contains("Delivered"), 1).otherwise(0)).withColumn("total_revenue", col("Amount") * col("Qty"))

print("=== BUSINESS PERFORMANCE ANALYSIS ===")

# 2. Category performance
category_performance = df_amazon_clean.groupBy("Category").agg(
count("*").alias("total_orders"),sum("total_revenue").alias("revenue"),avg("Amount").alias("avg_price"),(sum("is_delivered") / count("*") * 100).alias("delivery_rate")).filter(col("total_orders") >= 50).orderBy(col("revenue").desc())

print("Top Categories by Revenue:")
category_performance.show(10)

# 3. Monthly trend analysis
monthly_trend = df_amazon_clean.withColumn("order_month", date_format(to_date(col("Date"), "MM-dd-yy"), "yyyy-MM")).groupBy("order_month").agg(count("*").alias("orders"),sum("total_revenue").alias("monthly_revenue"),avg("Amount").alias("avg_order_value")).orderBy("order_month")

print("\nMonthly Sales Trend:")
monthly_trend.show()

# 4. Inventory analysis
print("\n=== INVENTORY ANALYSIS ===")
inventory_status = df_sales.withColumn("stock_status",when(col("Stock") == 0, "Out of Stock").when(col("Stock") < 3, "Low Stock").when(col("Stock") < 6, "Medium Stock").otherwise("High Stock")).groupBy("Category", "stock_status").count().orderBy("Category", col("count").desc())

inventory_status.show()

# 1. Revenue analysis
total_revenue = df_amazon_clean.agg(sum("total_revenue")).collect()[0][0]
total_orders = df_amazon_clean.count()
avg_order_value = df_amazon_clean.agg(avg("Amount")).collect()[0][0]

print(f"Total Revenue:\t{total_revenue:,.2f}")
print(f"Total Orders: {total_orders:,}")
print(f"Average Order Value:\t{avg_order_value:.2f}")

# 2. Top performing categories
top_3_categories = category_performance.select("Category", "revenue").take(3)
print(f"\nTop 3 Categories:")
for i, row in enumerate(top_3_categories, 1):
  print(f"{i}. {row['Category']}: {row['revenue']:,.2f}")

# 3. Delivery performance
overall_delivery_rate = df_amazon_clean.agg(
  (sum("is_delivered") / count("*") * 100).alias("delivery_rate")
).collect()[0][0]

print(f"\nOverall Delivery Success Rate: {overall_delivery_rate:.1f}%")

# 4. Inventory insights
low_stock_count = df_sales.filter(col("Stock") < 3).count()
out_of_stock_count = df_sales.filter(col("Stock") == 0).count()

print(f"\nInventory Status:")
print(f"Products with low stock: {low_stock_count}")
print(f"Out of stock products: {out_of_stock_count}")
