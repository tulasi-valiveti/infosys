# Databricks notebook source
# MAGIC %md
# MAGIC DayOfWeek → Day of the week (1 = Monday, …, 7 = Sunday).
# MAGIC
# MAGIC Date → Flight date.
# MAGIC
# MAGIC DepTime → Actual departure time (local, HHMM format, e.g., 1345 = 1:45 PM).
# MAGIC
# MAGIC ArrTime → Actual arrival time (local, HHMM format).
# MAGIC
# MAGIC CRSArrTime → Scheduled arrival time (as per airline’s published schedule).
# MAGIC
# MAGIC UniqueCarrier → Carrier code (e.g., "AA" = American Airlines, "DL" = Delta).
# MAGIC
# MAGIC Airline → Airline name (full name of UniqueCarrier).
# MAGIC
# MAGIC FlightNum → Flight number assigned by the airline.
# MAGIC
# MAGIC TailNum → Aircraft registration number (plane’s unique ID).
# MAGIC
# MAGIC ActualElapsedTime → Actual elapsed flight time in minutes (ArrTime − DepTime).
# MAGIC
# MAGIC CRSElapsedTime → Scheduled elapsed flight time in minutes.
# MAGIC
# MAGIC AirTime → Actual time spent flying in minutes (excluding taxiing).
# MAGIC
# MAGIC ArrDelay → Arrival delay in minutes (early arrivals = negative values).
# MAGIC
# MAGIC DepDelay → Departure delay in minutes.
# MAGIC
# MAGIC Origin → Origin airport code (e.g., "ATL" = Atlanta).
# MAGIC
# MAGIC Org_Airport → Origin airport full name (if available).
# MAGIC
# MAGIC Dest → Destination airport code (e.g., "LAX" = Los Angeles).
# MAGIC
# MAGIC Dest_Airport → Destination airport full name (if available).
# MAGIC
# MAGIC Distance → Distance between origin and destination airports (miles).
# MAGIC
# MAGIC TaxiIn → Taxi-in time in minutes (arrival gate from runway).
# MAGIC
# MAGIC TaxiOut → Taxi-out time in minutes (departure runway from gate).
# MAGIC
# MAGIC Cancelled → 1 = Flight cancelled, 0 = Not cancelled.
# MAGIC
# MAGIC CancellationCode → Reason for cancellation:
# MAGIC
# MAGIC "A" = Carrier
# MAGIC
# MAGIC "B" = Weather
# MAGIC
# MAGIC "C" = NAS (National Airspace System)
# MAGIC
# MAGIC "D" = Security
# MAGIC
# MAGIC Diverted → 1 = Flight diverted, 0 = Not diverted.
# MAGIC
# MAGIC CarrierDelay → Delay minutes due to airline (maintenance, crew, etc.).
# MAGIC
# MAGIC WeatherDelay → Delay minutes due to weather.
# MAGIC
# MAGIC NASDelay → Delay minutes due to National Airspace System (air traffic control, heavy traffic, etc.).
# MAGIC
# MAGIC SecurityDelay → Delay minutes due to security (e.g., evacuation, screening).
# MAGIC
# MAGIC LateAircraftDelay → Delay minutes because the aircraft arrived late from a previous flight.

# COMMAND ----------

import pandas as pd

# Replace 'your_file.csv' with the path to your dataset
data = pd.read_csv("/Volumes/workspace/default/airlines/Flight_delay.csv")

# Display first 5 rows
print(data.head())


# COMMAND ----------





# Get all column names
columns = data.columns.tolist()

print("Columns in the dataset:")
print(columns)


# COMMAND ----------

print(data.head())

# COMMAND ----------

# Set Pandas option to display ALL columns
pd.set_option('display.max_columns', None)

# Now info() and describe() will show everything
print("\n--- Dataset Info ---")
data.info()




# COMMAND ----------

print("\n--- Dataset Description ---")
print(data.describe(include='all'))  # include='all' adds categorical columns too

# COMMAND ----------

# Remove duplicate rows
data_no_duplicates = data.drop_duplicates()
print("Before removing duplicates:", data.shape)
print("After removing duplicates:", data_no_duplicates.shape)

# COMMAND ----------

# ---- General Missing Value Check ----
print("Missing values per column:")
print(data.isnull().sum())


# COMMAND ----------

# ---- Handling Missing Values ----

# 1. For categorical text fields → fill with "Unknown" or most frequent value
categorical_cols = ["Org_Airport", "Dest_Airport", "CancellationCode", "TailNum"]
for col in categorical_cols:
    if col in data.columns:
        data[col] = data[col].fillna("Unknown")   # or use df[col].mode()[0]

# COMMAND ----------

# 2. For numeric fields related to delays → replace NaN with 0 (means no delay recorded)
delay_cols = ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
for col in delay_cols:
    if col in data.columns:
        data[col] = data[col].fillna(0)

# COMMAND ----------

# 3. For elapsed time or airtime → fill with median (less skewed than mean)
time_cols = ["ActualElapsedTime", "AirTime"]
for col in time_cols:
    if col in data.columns:
        data[col] = data[col].fillna(data[col].median())

# COMMAND ----------

# ---- Cancellation column ----
# If 'CancellationCode' is null, it means the flight was NOT cancelled
if "CancellationCode" in data.columns:
    data["CancellationCode"] = data["CancellationCode"].fillna("Not Cancelled")

# COMMAND ----------

# ---- Check results ----
print(data[delay_cols + ["Cancelled", "CancellationCode"]].isnull().sum())

# COMMAND ----------

# 4. If still missing values remain → drop those rows (safe clean-up)
df = data.dropna()

# COMMAND ----------

# ---- Verify again ----
print("\nMissing values after cleaning:")
print(data.isnull().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC Delay columns → Null means no delay logged → safely replaced with 0.
# MAGIC
# MAGIC CancellationCode → Null means flight wasn’t cancelled → replaced with "Not Cancelled".
# MAGIC
# MAGIC Cancelled column (0/1) → already numeric, but you can also double-check consistency (if Cancelled=0, then CancellationCode should be "Not Cancelled").

# COMMAND ----------

# Save cleaned dataset
cleaned_dataset=data.to_csv("flights_cleaned.csv", index=False)

# COMMAND ----------

print(cleaned_dataset)

# COMMAND ----------

csv_string = data.to_csv(index=False)
print(csv_string)  # now csv_string contains the CSV text


# COMMAND ----------

# ---- Convert 'Date' + 'DepTime' into a proper datetime ----
# Step 1: Ensure Date is datetime
data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")



# COMMAND ----------

# Step 2: Convert DepTime (HHMM int) into string with leading zeros
data["DepTime"] = data["DepTime"].apply(lambda x: f"{int(x):04d}" if pd.notnull(x) else "0000")


# COMMAND ----------

# Step 3: Extract hour + minute
data["DepHour"] = data["DepTime"].str[:2].astype(int)
data["DepMinute"] = data["DepTime"].str[2:].astype(int)

# COMMAND ----------

# Step 4: Combine Date + Time into full datetime
df["DepDatetime"] = df["Date"] + pd.to_timedelta(df["DepHour"], unit="h") + pd.to_timedelta(df["DepMinute"], unit="m")


# COMMAND ----------

# ---- Derived Features ----
df["Month"] = df["DepDatetime"].dt.month          # Month number (1–12)
df["DayOfWeek"] = df["DepDatetime"].dt.day_name() # Monday, Tuesday...
df["Hour"] = df["DepDatetime"].dt.hour            # Hour of departure
df["Route"] = df["Origin"] + "-" + df["Dest"]     # Route string (e.g., ATL-LAX)

# COMMAND ----------

# ---- Check results ----
print(df[["Date", "DepTime", "DepDatetime", "Month", "DayOfWeek", "Hour", "Route"]].head())