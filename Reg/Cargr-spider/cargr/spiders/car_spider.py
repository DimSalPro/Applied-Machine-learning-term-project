import scrapy
from ..items import CargrItem
import time
import random
import pandas as pd



class QuoteSpider(scrapy.Spider):
    # this script will call all dependencies when needed
    # in the terminal type " scrapy crawl cars -o <name_of_the_file.csv>",
    # and it automatily visit the website and extract the specified information

    name = 'cars'
    page_num = 2
    start_urls = [
        f'https://www.car.gr/classifieds/cars/?condition=used&offer_type=sale&pg=1&price-from=%3E500&registration-from=%3E2000&significant_damage=f'
    ]

    def edit_raw_text(self, word):
        # This function will remove punctuations before storing
        # Clean data before save
        word = word.replace('\\xa0€', '')
        word = word.replace('\n', '')
        word = word.replace(' ', '')
        word = word.replace('\xa0€', '')
        word = word.replace('`', '')
        word = word.replace('cc', '')
        word = word.replace('bhp', '')
        word = word.replace('χλμ', '')
        word = word.replace('\xa0', '')
        word = word.replace(',', '')
        word = word.replace('.', '')
        self.transmission_conv(word)
        return word

    def transmission_conv(self,word):
        # Tranformation of field car type
        word = word.replace("Αέριο(lpg)-Βενζίνη", 'gas')
        word = word.replace("Φυσικόαέριο(cng)", 'gas')
        word = word.replace('ΥβριδικόΠετρέλαιο', 'hybrid')
        word = word.replace('ΥβριδικόΒενζίνη', 'hybrid')
        word = word.replace('Βενζίνη', 'gasoline')
        word = word.replace('Πετρέλαιο', 'diesel')
        word = word.replace('Ηλεκτρικό', 'electric')
        return word

    def parse(self, response, **kwargs):
        # Items refers to class items, where the fields where defined for temporary storing
        items = CargrItem()

        maker =response.xpath('//div[@itemprop="title"]/text()')[::2].extract()
        model = response.xpath('//span[@itemprop="model"]/text()')[::2].extract()
        year = response.xpath('//span[@itemprop="releaseDate"]/text()').extract()
        price = response.xpath('//span[@itemprop="price"]/text()').extract()
        car_type = response.xpath('//span[@class="text-muted test font-weight-bold"]/text()').extract()
        mileage = response.xpath('//span[@span="Mileage"]/text()')[1::2].extract()
        engine = response.xpath('//span[@class="engine colorize"]/text()')[1::2].extract()
        transmission = response.xpath('//span[@class="transmision colorize"]/text()')[1::2].extract()
        car_fuel = response.xpath('//span[@class="fueltype colorize"]/text()')[1::2].extract()
        link = response.xpath('//div[@class="clsfd_list_row"]/a/@href').extract()

        for i in range(len(response.css('div.clsfd_list_row_group'))):
            try:
                items['maker'] = self.edit_raw_text(maker[i])
                items['model'] = self.edit_raw_text(model[i])
                items['year'] = self.edit_raw_text(year[i])
                items['price'] = self.edit_raw_text(price[i])
                items['car_type'] = self.edit_raw_text(car_type[i])
                items['mileage'] = self.edit_raw_text(mileage[i])

                items['transmission'] = self.edit_raw_text(transmission[i])
                items['car_fuel'] = self.edit_raw_text(car_fuel[i])
                items['link'] = link[i]

                # Splitting field engine to cc and hp
                try:
                    items['hp'] = self.edit_raw_text(engine[i].split('/')[1])
                except:
                    items['hp'] = 0

                try:
                    items['cc'] = self.edit_raw_text(engine[i].split('/')[0])
                except:
                    items['cc'] =0

                # with open('data.csv','a') as f:
                #     writer = f"""{items['maker']},{items['price']}\n"""
                #     f.write(writer)
                # f.close()
                yield items
            except IndexError:
                pass


        next_page = f'https://www.car.gr/classifieds/cars/?condition=used&offer_type=sale&pg={str(QuoteSpider.page_num)}&price-from=%3E500&registration-from=%3E2000&significant_damage=f'
        # Replace the number bellow to the maximum page of the website
        if int(QuoteSpider.page_num) < 4030:
            QuoteSpider.page_num += 1
            time.sleep(random.randint(0, 5))
            yield response.follow(next_page, callback=self.parse)

