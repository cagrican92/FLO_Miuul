

###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranış öbeklenmelerine göre gruplar oluşturulacak..

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

###############################################################
# GÖREVLER
###############################################################

# GÖREV 1: Veriyi Anlama (Data Understanding) ve Hazırlama
           # 1. flo_data_20K.csv verisini okuyunuz.
import pandas as pd

from datetime import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' %x )
pd.set_option('display.width',1000)
df_= pd.read_csv('Odevler/flo_data_20k.csv')
df = df_.copy()

           # 2. Veri setinde
                     df.shape
                     # a. İlk 10 gözlem,
                     df.head(10)

                     df.groupby['master_id'].agg({''})
                     # b. Değişken isimleri,
                     df.columns
                     # c. Betimsel istatistik,
                    df.describe().T
                     # d. Boş değer,
                    df.isnull().sum().sum()
                     # e. Değişken tipleri, incelemesi yapınız.
               df.dtypes
               type("first_order_date")

           # 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
           # alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

df["total_order_number_ever"]= df["order_num_total_ever_online"] +  df["order_num_total_ever_offline"]
df["total_costumer_value_ever"] = df["customer_value_total_ever_offline"] +  df["customer_value_total_ever_online"]

           # 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.


for columbs in df.columns:
    if columbs.__contains__("date") == True:
        df[columbs] = pd.to_datetime(df[columbs])
        print( str(columbs) + "  :  " +  str(df[columbs].dtypes))


           # 5. Alışveriş kanallarındaki müşteri sayısının, ortalama alınan ürün sayısının ve ortalama harcamaların dağılımına bakınız.

df.groupby("order_channel").agg({"master_id": "count",
                                "total_order_number_ever": "mean",
                                "total_costumer_value_ever": "mean"})


           # 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.

df.groupby("master_id").agg({"total_costumer_value_ever": "sum"}).sort_values( "total_costumer_value_ever" ,ascending= False).head(10)

           # 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
df.groupby("master_id").agg({"total_order_number_ever": "sum"}).sort_values( "total_order_number_ever" ,ascending= False).head(10)
           # 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.
# GÖREV 2: RFM Metriklerinin Hesaplanması


from datetime import datetime, timedelta

df["last_order_date"].max()

today = df["last_order_date"].max() + timedelta(days = 2)

rfm = df.groupby("master_id").agg({ 'last_order_date': lambda date: (today - date.max()).days,
                              'total_order_number_ever': lambda total_order_number_ever: total_order_number_ever.sum(),
                                'total_costumer_value_ever': lambda total_costumer_value_ever: total_costumer_value_ever.sum()})
df.loc[ df['master_id'] == '00016786-2f5a-11ea-bb80-000d3a38a36f']
rfm.columns = ['recency', 'frequency', 'monetary']

rfm.describe().T

# GÖREV 3: RF ve RFM Skorlarının Hesaplanması

rfm['recency_score'] = pd.qcut(rfm['recency'],5, labels=[5,4,3,2,1])
rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'),5, labels=[1,2,3,4,5])
rfm['monetary_score'] = pd.qcut(rfm['monetary'],5, labels=[1,2,3,4,5])

rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması

segmentasyon = {
r'[1-2][1-2]': 'hipernating',
r'[1-2][3-4]': 'at_Risk',
r'[1-2]5': 'cant_loose',
r'3[1-2]': 'about_to_sleep',
r'33': 'need_attention',
r'[3-4][4-5]': 'loyal_customer',
r'41': 'promising',
r'51': 'new_customer',
r'[4-5][2-3]': 'potential_loyalist',
r'5[4-5]': 'champions'
 }

rfm["Segment"] = rfm["RFM_SCORE"].replace(segmentasyon, regex= True)

# GÖREV 5: Aksiyon zamanı!
           # 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
           rfm.groupby(['Segment']).agg({'recency': 'mean',
                                         'frequency': 'mean',
'monetary': 'mean'})
           # 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv ye kaydediniz.

                   # a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
                   # tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Sadık müşterilerinden(champions,loyal_customers),
                   # ortalama 250 TL üzeri ve kadın kategorisinden alışveriş yapan kişiler özel olarak iletişim kuralacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına
                   # yeni_marka_hedef_müşteri_id.cvs olarak kaydediniz.

 loyal_and_champions = rfm[(rfm['Segment']=='champions')].index
 loyal_and_champions = rfm[(rfm['Segment']=='loyal_customer') ].index.append(loyal_and_champions)
type(loyal_and_champions)

kadın_and_250 = df[(df["interested_in_categories_12"].str.contains('KADIN') == True) & (df["total_costumer_value_ever"] > 250)]['master_id']


potantial_new_brand_costumer =  loyal_and_champions.intersection(kadın_and_250)
dfx = pd.DataFrame(index =potantial_new_brand_costumer )
dfx.to_csv("yeni_marka_hedef_müşteri_id.csv")

                   # b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan ama uzun süredir
                   # alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
                   # olarak kaydediniz.


# GÖREV 6: Tüm süreci fonksiyonlaştırınız.
