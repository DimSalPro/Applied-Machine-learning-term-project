# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class CargrItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    maker = scrapy.Field()
    model = scrapy.Field()
    year = scrapy.Field()
    price = scrapy.Field()
    car_type = scrapy.Field()
    mileage = scrapy.Field()
    # engine = scrapy.Field()
    hp = scrapy.Field()
    cc = scrapy.Field()
    transmission = scrapy.Field()
    car_fuel = scrapy.Field()
    link = scrapy.Field()

