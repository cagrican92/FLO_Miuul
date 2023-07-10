##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.


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
# GÖREV 1: Veriyi Hazırlama
           # 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
#pip install lifetimes
import lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width',1000)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler


df_ = pd.read_csv('Odevler/flo_data_20k.csv')
df = df_.copy()
df.isnull().sum()
df.head()
           # 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
           # Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.

def outlier_thresholds (dataframe, veriable):
    quartile1 = dataframe[veriable].quantile(0.01)
    quartile3 = dataframe[veriable].quantile(0.99)
    interquantile_range = quartile3-quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 -1.5 * interquantile_range
    return low_limit.round(), up_limit.round()

def replace_with_threshold(dataframe,veriable):
    low_limit, up_limit = outlier_thresholds(dataframe,veriable)
    dataframe.loc[(dataframe[veriable] > up_limit), veriable] = up_limit.round()

           # 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
           # aykırı değerleri varsa baskılayanız.
    outlier_thresholds(df,"order_num_total_ever_online" )
    replace_with_threshold(df,"order_num_total_ever_online")
    outlier_thresholds(df, "order_num_total_ever_offline")
    replace_with_threshold(df, "order_num_total_ever_offline")
    outlier_thresholds(df, "customer_value_total_ever_offline")
    replace_with_threshold(df, "customer_value_total_ever_offline")
    outlier_thresholds(df, "customer_value_total_ever_online")
    replace_with_threshold(df, "customer_value_total_ever_online")
           # 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
           # alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

    df['total_order_num_total_ever'] = df['order_num_total_ever_offline'] + df['order_num_total_ever_online']
    df['total_customer_value_total_ever'] = df['customer_value_total_ever_online'] + df['customer_value_total_ever_offline']

    df[df['total_order_num_total_ever'] < 1]
    df[df['total_customer_value_total_ever'] < 1]
    # 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
for columbs in df.columns:
    if columbs.__contains__("date") == True:
        df[columbs] = pd.to_datetime(df[columbs])
        print( str(columbs) + "  :  " +  str(df[columbs].dtypes))

df['date_diff'] = (df['last_order_date'] - df['first_order_date']).dt.days
df[df['date_diff'] < 1]

df = df[~df['date_diff'] < 1]
# GÖREV 2: CLTV Veri Yapısının Oluşturulması

           # 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
from datetime import datetime, time, timedelta
df["last_order_date"].max()
today = df["last_order_date"].max() + timedelta(days = 2)
           # 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
cltv_df = df.groupby('master_id').agg({'date_diff': lambda  date_diff: date_diff ,
                                       'first_order_date': lambda first_order_date: (today - first_order_date.min()).days,
                                       'total_order_num_total_ever': lambda total_order_num_total_ever: total_order_num_total_ever.sum(),
                                       'total_customer_value_total_ever': lambda total_customer_value_total_ever: total_customer_value_total_ever.sum()})
cltv_df.reset_index()
#cltv_df.columns = cltv_df.columns.droplevel(0)

cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
cltv_df['monetary'] = cltv_df['monetary'] / cltv_df['frequency']
cltv_df.describe().T
cltv_df = cltv_df[cltv_df['frequency'] > 1 ]

cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7


           # Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.
 #üstte

# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması

           # 1. BG/NBD modelini fit ediniz.
from lifetimes import BetaGeoFitter
bgf = BetaGeoFitter(penalizer_coef=0.001)
#problem var sor!!!
bgf.fit(cltv_df['frequency'], cltv_df['recency'], cltv_df['T'])

                # a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
bgf.conditional_expected_number_of_purchases_up_to_time(12,cltv_df['frequency'],cltv_df['recency'], cltv_df['T']).sort_values(ascending=False).head(10)
cltv_df['expected_purc_3_mount'] = bgf.conditional_expected_number_of_purchases_up_to_time(12,cltv_df['frequency'], cltv_df['recency'] ,cltv_df['T'])
                # b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
bgf.conditional_expected_number_of_purchases_up_to_time(24,cltv_df['frequency'],cltv_df['recency'], cltv_df['T']).sort_values(ascending=False).head(10)
cltv_df['expected_purc_6_mount'] = bgf.conditional_expected_number_of_purchases_up_to_time(24,cltv_df['frequency'],cltv_df['recency'], cltv_df['T'])
plot_period_transactions(bgf)
plt.show(block=True)
           # 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
ggf = GammaGammaFitter(penalizer_coef=0.001)
ggf.fit(cltv_df['frequency'],cltv_df['monetary'])
ggf.conditional_expected_average_profit(cltv_df['frequency'], cltv_df['monetary']).sort_values(ascending = False).head(10)
cltv_df['expected_avarage_profit'] = ggf.conditional_expected_average_profit(cltv_df['frequency'], cltv_df['monetary'])
           # 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
cltv = ggf.customer_lifetime_value(bgf,cltv_df['frequency'],cltv_df['recency'],cltv_df['T'],cltv_df['monetary'], time = 6, freq='W',discount_rate=0.01)
                # b. Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.
cltv.head(20)
cltv = cltv.reset_index()

cltv_final = cltv_df.merge(cltv,on = 'master_id', how = 'left')

# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
cltv_final['segment'] = pd.qcut(cltv_final['clv'],4,labels=['D','C','B','A'])

           # 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. cltv_segment ismi ile dataframe'e ekleyiniz.

           # 2. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz

# BONUS: Tüm süreci fonksiyonlaştırınız.









