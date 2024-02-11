from pyspark.sql import SparkSession
from pyspark.sql.functions import col, coalesce, udf, broadcast
from pyspark.sql.types import StringType, StructType, StructField, DoubleType
import Geohash
import requests
import zipfile
import os
import pandas as pd


def get_coordinates(name, country, city):
    query = f"{name}, {country}, {city}"
    url = f'https://api.opencagedata.com/geocode/v1/json?q={query}&key=72a9bba18608437f9a72109fdde7b0ea'
    response = requests.get(url)
    data = response.json()
    if data['results']:
        lat = data['results'][0]['geometry']['lat']
        lng = data['results'][0]['geometry']['lng']
        return lat, lng
    return None, None


def generate_geohash(lat, lng):
    try:
        return Geohash.encode(lat, lng, precision=4)
    except Exception as e:
        return None


def unzip_files(zip_folder, destination_folder):
    for filename in os.listdir(zip_folder):
        if filename.endswith('.zip'):
            zip_path = os.path.join(zip_folder, filename)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for elem in zip_ref.namelist():
                    if elem.startswith('weather/'):
                        zip_ref.extract(elem, destination_folder)


if __name__ == '__main__':
    # Creating a Spark session
    spark = SparkSession.builder \
        .master('local[*]') \
        .appName("epam_spark_hw") \
        .getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    # Reading restaurant data
    df = spark.read.csv("data/restaurant_csv/", header=True, inferSchema=True)

    # Filtering restaurant data to get only rows
    # with null values in columns latitude and longitude
    df_null_coordinates = df.filter(col('lat').isNull() | col('lng').isNull())

    # Get coordinates for each place
    # Using Pandas for ease of iteration, could use UDF as well
    df_null_coordinates_pandas = df_null_coordinates.toPandas()
    for index, row in df_null_coordinates_pandas.iterrows():
        lat, lng = get_coordinates(row['franchise_name'], row['country'], row['city'])
        df_null_coordinates_pandas.at[index, 'lat'] = lat
        df_null_coordinates_pandas.at[index, 'lng'] = lng

    # # Apply the get_coordinates function to each row using apply
    # df_null_coordinates_pandas[['lat', 'lng']] = df_null_coordinates_pandas.apply(
    #     lambda row: pd.Series(get_coordinates(row['franchise_name'], row['country'], row['city'])),
    #     axis=1
    # )


# Create DataFrame with updated coordinates
    updated_coordinates_df = spark.createDataFrame(df_null_coordinates_pandas)

    # Join updated coordinates with the original DataFrame
    df_final = df.join(updated_coordinates_df, ['id'], 'left_outer')

    # Select final columns and apply coalesce for lat and lng
    df_final = df_final.select(
        df['id'],
        coalesce(updated_coordinates_df['lat'], df['lat']).alias('lat'),
        coalesce(updated_coordinates_df['lng'], df['lng']).alias('lng'),
        *[df[col] for col in df.columns if col not in ['id', 'lat', 'lng']]
    )

    # Define a UDF for generating geohash
    generate_geohash_udf = udf(generate_geohash, StringType())

    # Add geohash column to the DataFrame
    df_with_geohash = df_final.withColumn("Geohash", generate_geohash_udf("lat", "lng"))

    # Unpack weather dataset from zip files
    unzip_files('data/weather/', 'data/weather/all/')

    # Read weather DataFrame
    schema = StructType([
        StructField("lat", DoubleType(), True),
        StructField("lng", DoubleType(), True),
        StructField("year", StringType(), True),
        StructField("month", StringType(), True),
        # Add other fields from your weather schema
    ])
    df_weather = spark.read.format("parquet").schema(schema).load("data/weather/all/")

    # Add geohash column to the weather DataFrame
    df_weather_with_geohash = df_weather.withColumn("Geohash", generate_geohash_udf("lat", "lng"))

    # Drop lat and lng from restaurant DataFrame
    df_with_geohash_cut = df_with_geohash.drop('lat', 'lng')

    # Join DataFrames with broadcast for optimization
    df_joined = df_weather_with_geohash.join(broadcast(df_with_geohash_cut), "geohash", "left")

    # Repartition to optimize performance and memory usage
    df_repartitioned = df_joined.repartition("year", "month")

    # Write the final DataFrame to parquet
    df_repartitioned.write.partitionBy("year", "month").parquet('refined/weather_and_restaurants')

    # Stopping the session
    spark.stop()

