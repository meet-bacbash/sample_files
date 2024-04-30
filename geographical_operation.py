import asyncio
from concurrent.futures import ThreadPoolExecutor
from .geo_location_details import get_address_details
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession


NO_REQUEST_THREADS = 32


async def generate_geo_columns(address_list: list) -> tuple:
    """
    Request geo API asynchronously.
    :param address_list: List of geographical fields (city, state, country, coordinates)
    :return: List of results from geo API.
    """
    print("Total locations to be fetched:", len(address_list))
    with ThreadPoolExecutor(NO_REQUEST_THREADS) as exe:
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(exe, get_address_details, address) for address in address_list]
        results = await asyncio.gather(*tasks)
    return results


def get_address_detail_for_distinct_values(dataframe: DataFrame, input_column: str, country_code: str) -> tuple:
    """
    Fetch detailed addresses for distinct values of input column and country code.
    :param dataframe: Input DataFrame.
    :param input_column: Column containing zip code/address data.
    :param country_code: Column containing country code.
    :return: List of detail addresses.
    """

    # Create list of distinct values for input column
    unique_address_list = dataframe.dropDuplicates([input_column, country_code]).select(input_column, country_code)
    unique_address_list = [tuple(row) for row in unique_address_list.collect()]

    # Fetch detailed address for all distinct pincode,streets
    detail_addresses = asyncio.run(
        generate_geo_columns(
            address_list=unique_address_list
        )
    )
    return detail_addresses


# Test (locla):
if __name__ == "__main__":
    spark = SparkSession.builder.master("local").appName("AddressDetails").getOrCreate()
    df = spark.read.csv("Mack_dataset.csv", header=True, inferSchema=True)
    addresses = get_address_detail_for_distinct_values(df, "address_column", "country_code_column")
