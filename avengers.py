import numpy as np
import pandas as pd

from math import log, sqrt, pi, exp
from scipy.stats import norm

import matplotlib.pyplot as plt

#тут необходимо добавить более ранние серии и отработать 3/22
EXP_DATES = {
    '3.17' : {
    'qrt' : ['160317'],
    'mnth' : ['160317', '160217', '190117'],
    'wk' : ['090317', '020317', '220217', '090217']
    },
    '6.17' : {
    'qrt' : ['150617'],
    'mnth' : ['150617', '180517', '200417'],
    'wk' : ['080617', '010617', '250517', '110517', '040517', '270417', '130417', '060417', '300317', '230317']
    },
    '9.17' : {
    'qrt' : ['210917'],
    'mnth' : ['210917', '170817', '200717'],
    'wk' : ['140917', '070917', '310817', '240817', '100817', '030817', '270717', '130717', '060717', '290617', '220617']
    },
    '12.17' : {
    'qrt' : ['211217'],
    'mnth' : ['211217', '161117', '191017'],
    'wk' : ['141217', '071217', '301117', '231117', '091117', '021117', '261017', '121017', '051017', '280917']
    },
    '3.18' : {
    'qrt' : ['150318'],
    'mnth' : ['150318', '150218', '180118'],
    'wk' : ['070318', '010318', '220218', '080218', '010218', '250118', '110118', '281217']
    },
    '6.18' : {
    'qrt' : ['210618'],
    'mnth' : ['210618', '170518', '190418'],
    'wk' : ['140618', '070618', '310518', '240518', '100518', '030518', '260418', '120418', '050418', '290318', '220318']
    },
    '9.18' : {
    'qrt' : ['200918'],
    'mnth' : ['200918', '160818', '190718'],
    'wk' : ['130918', '060918', '300818', '230818', '090818', '020818', '260718', '120718', '050718', '280618']
    },
    '12.18' : {
    'qrt' : ['201218'],
    'mnth' : ['201218', '151118', '181018'],
    'wk' : ['131218', '061218', '291118', '221118', '081118', '011118', '251018', '111018', '041018', '270918']
    },
    '3.19' : {
    'qrt' : ['210319'],
    'mnth' : ['210319', '210219', '170119'],
    'wk' : ['140319', '070319', '280219', '140219', '070219', '310119', '240119', '100119', '271219']
    },
    '6.19' : {
    'qrt' : ['200619'],
    'mnth' : ['200619', '160519', '180419'],
    'wk' : ['130619', '060619', '300519', '230519', '080519', '020519', '250419', '110419', '040419', '280319']
    },
    '9.19' : {
    'qrt' : ['190919'],
    'mnth' : ['190919', '150819', '180719'],
    'wk' : ['120919', '050919', '290819', '220819', '080819', '010819', '250719', '110719', '040719', '270619']
    },
    '12.19' : {
    'qrt' : ['191219'],
    'mnth' : ['191219', '211119', '171019'],
    'wk' : ['121219', '051219', '281119', '141119', '071119', '311019', '241019', '101019', '031019', '260919']
    },
    '3.20' : {
    'qrt' : ['190320'],
    'mnth' : ['190320', '200220', '160120'],
    'wk' : ['120320', '050320', '270220', '130220', '060220', '300120', '230120', '090120', '261219']
    },
    '6.20' : {
    'qrt' : ['180620'],
    'mnth' : ['180620', '210520', '160420'],
    'wk' : ['110620', '040620', '280520', '140520', '070520', '300420', '230420', '090420', '020420', '260320']
    },
    '9.20' : {
    'qrt' : ['170920'],
    'mnth' : ['170920', '200820', '160720'],
    'wk' : ['100920', '030920', '270820', '130820', '060820', '300720', '230720', '090720', '020720', '250620']
    },
    '12.20' : {
    'qrt' : ['171220'],
    'mnth' : ['171220', '191120', '151020'],
    'wk' : ['101220', '031220', '261120', '121120', '051120', '291020', '221020', '081020', '011020', '240920']
    },
    '3.21' : {
    'qrt' : ['180321'],
    'mnth' : ['180321', '180221', '210121'],
    'wk' : ['110321', '040321', '250221', '110221', '040221', '280121', '140121', '060121', '301220', '241220']
    },
    '6.21' : {
    'qrt' : ['170621'],
    'mnth' : ['170621', '200521', '150421'],
    'wk' : ['100621', '030621', '270521', '130521', '060521', '290421', '220421', '080421', '010421', '250321']
    },
    '9.21' : {
    'qrt' : ['160921'],
    'mnth' : ['160921', '190821', '150721'],
    'wk' : ['090921', '020921', '260821', '120821', '050821', '290721', '220721', '080721', '010721', '240621']
    },
    '12.21' : {
    'qrt' : ['161221'],
    'mnth' : ['161221', '181121', '211021'],
    'wk' : ['091221', '021221', '251121', '111121', '031121', '281021', '141021', '071021', '300921', '230921']
    },
    '6.22' : {
    'qrt' : ['160622'],
    'mnth' : ['160622', '190522', '210422'],
    'wk' : ['090622', '020622', '260522', '120522', '050522', '280422', '140422', '070422']
    },
    '9.22' : {
    'qrt' : ['150922'],
    'mnth' : ['150922', '180822', '210722'],
    'wk' : ['080922', '010922', '250822', '110822', '040822', '280722', '140722', '070722', '300622', '230622']
    },
    '12.22' : {
    'qrt' : ['151222'],
    'mnth' : ['151222', '171122', '201022'],
    'wk' : ['081222', '011222', '241122', '101122', '031122', '271022', '131022', '061022', '290922', '220922']
    },
    '3.23' : {
    'qrt' : ['160323'],
    'mnth' : ['160323', '160223', '190123'],
    'wk' : ['221222', '291222', '050123']
    }
}

TARGET_COLUMN_NAME = 'adjp'

class Jarvis:
    """
    Jarvis class for fun and profit.
    
    This class contains some data management, futures and options analisys features.
    Backtesting is planned.
    """
    
    def __init__(self, name, fut = True, opt = True, iv = True):
        
        #-----init_constants:
        self.N_STRIKES = 8
        self.FUTCODES_AND_STRIKESTEPS = {
            'RTS': 2500,
            'Si': 500,
            'GAZR': 250,
            'SBRF': 250
        }
        self.MONEYNESS_STRIKES = {
            'minus_five' : -5, 'minus_four' : -4, 'minus_three' : -3, 'minus_two' : -2, 'minus_one' : -1,
            'center': 0,
            'plus_one' : +1, 'plus_two' : +2, 'plus_three' : +3, 'plus_four' : +4, 'plus_five' : +5
        }
        self.CALL_PUT = {'C' : 'call', 'P' : 'put'}
        self.PATH_WEB_1 = \
        'https://www.moex.com/ru/derivatives/contractresults-exp.aspx?day1=20000101&day2=20230316&code='
        self.PATH_WEB_2 = \
        'https://www.moex.com/ru/derivatives/contractresults-exp.aspx?type=2&day1=20000101&day2=20230316&code='
        self.PATH_LOCAL_FUTURES = 'FUTURES/'
        self.PATH_LOCAL_OPTIONS = 'OPTIONS/'
        self.COLUMNS = ['date', 'wavp', 'adjp', 'open', 'high', 'low', 'close', 'chng', 'lt_vol', 
                        'n_trades', 'vol_rub', 'vol_contr', 'oi_rub', 'oi_contr']
        self.TARGET_COLUMN = TARGET_COLUMN_NAME
        
        #-----init_variables:
        self.name = name
        split_ = self.name.split('-')
        self.key = split_[0]
        self.date = split_[1]
        self.strike_step = self.FUTCODES_AND_STRIKESTEPS[self.key]
        if fut:
            self.parse_fut_log, self.futures_df = self.fut()
        if opt:
            #в этом блоке необходимо добавить месячные и позднее недельные опционы в отдельных датафреймах
            self.exp_date = EXP_DATES[self.date]['qrt'][0]
            if self.key in ['GAZR', 'SBRF']:
                self.exp_date = str(int(self.exp_date) - 10000)
            self.lower = round(self.futures_df[self.TARGET_COLUMN].min() / self.strike_step) * self.strike_step
            self.upper = round(self.futures_df[self.TARGET_COLUMN].max() / self.strike_step) * self.strike_step
            self.qrt_opt_parse_log, self.qrt_opt = self.opt()
        if iv:
            self.vola_log, self.vola_df = self.center_strike()

    #-----B.-Sh. formulae implementation:
    def d1_calc(self, spot, strike, sigma, T):
        result = (log(spot/strike) + 0.5 * (sigma ** 2) * (T)) / (sigma * sqrt(T))
        return result

    def n_d1_calc(self, d1):
        result = (1 / sqrt(2 * pi)) * exp(-0.5 * (d1 ** 2))
        return result

    def price(self, spot, strike, sigma, T, option_type):
        d1 = self.d1_calc(spot, strike, sigma, T)
        d2 = d1 - sigma * sqrt(T)
        if option_type == 'call':
            N_d2 = norm.cdf(d2)
            delta_call = norm.cdf(d1)
            result = (spot * delta_call - strike * N_d2)
        if option_type == 'put':
            N_minus_d2 = norm.cdf(-d2)
            delta_put = (norm.cdf(d1) - 1)
            result = (strike * N_minus_d2 + spot * delta_put)
        return result

    def delta(self, spot, strike, sigma, T, option_type):
        d1 = self.d1_calc(spot, strike, sigma, T)
        delta_call = norm.cdf(d1)
        if option_type == 'call':
            result = delta_call
        if option_type == 'put':
            result = (delta_call - 1)
        return result

    def gamma(self, spot, strike, sigma, T):
        d1 = self.d1_calc(spot, strike, sigma, T)
        n_d1 = self.n_d1_calc(d1)
        result = n_d1 / (sigma * spot * sqrt(T))
        return result

    def theta(self, spot, strike, sigma, T):
        d1 = self.d1_calc(spot, strike, sigma, T)
        n_d1 = self.n_d1_calc(d1)
        result = (-spot * n_d1 * sigma / (2 * sqrt(T))) / 365
        return result

    def vega(self, spot, strike, sigma, T):
        d1 = self.d1_calc(spot, strike, sigma, T)
        n_d1 = self.n_d1_calc(d1)
        result = spot * sqrt(T) * n_d1 / 100
        return result

    #-----implied_volatility_iteration:
    def loss_func(self, spot, strike, T, option_type, sigma, premium):
        "Ф-ия потерь: в данном случае сравнением расчетной премии с фактической."
        return self.price(spot, strike, sigma, T, option_type) - premium

    def compute_iv(self, spot, strike, T, option_type, premium):
        "Функция итеративного поиска значения подразумеваемой волатильности."
        left = 0.01
        right = 10
        error = 0.0000001
        while right - left > error:
            if self.loss_func(spot, strike, T, option_type, left, premium) == 0:
                return left
            middle = (left + right) / 2
            if self.loss_func(spot, strike, T, option_type, left, premium) * \
            self.loss_func(spot, strike, T, option_type, middle, premium) < 0:
                right = middle
            else:
                left = middle
        return left

    #-----расчет подразумеваемых волатильностей для фактических значений текущих датафреймов
    def get_ivs(self, row):
        prem = row['center_prem']
        if np.isnan(prem):
            return np.NaN
        T = row['T']
        if T == 0:
            return np.NaN
        spot = row['spot']
        if np.isnan(spot):
            return np.NaN
        center = row['center']
        if np.isnan(center):
            return np.NaN
        try:
            x = self.compute_iv(spot, center, T, 'call', prem) * 100
        except:
            x = np.NaN
        return x
    
    def get_premium(self, row):
        date = row.name
        try:
            x = self.qrt_opt[(self.qrt_opt.index == date) &
                           (self.qrt_opt['strike'] == row['center']) &
                           (self.qrt_opt['option_type'] == 'call')][self.TARGET_COLUMN][0]
        except:
            x = np.NaN
        return x

    def center_strike(self):
        vola_log = {}
        vola_df = pd.DataFrame(index = self.futures_df.index)
        vola_df['T'] = ((pd.to_datetime(self.exp_date, format = '%d%m%y') - vola_df.index).days / 365)
        vola_df['spot'] = self.futures_df[self.TARGET_COLUMN]
        vola_df['center'] = round(vola_df['spot'] / self.strike_step) * self.strike_step
        vola_df['center_prem'] = vola_df.apply(self.get_premium, axis = 1)
        vola_df['center_iv'] = vola_df.apply(self.get_ivs, axis = 1)
        #vola_df = vola_df.dropna()
        return vola_log, vola_df
    
    #-----ниже блок загрузки цсв файлов
    def parse_csv(self, name, parse_from, datetime_format):
        "Парсим цсв (внезапно)."
        x = pd.DataFrame()
        result = {}
        result['name'] = name
        try:
            x = pd.read_csv(parse_from, encoding='cp866')
            try:
                x.columns = self.COLUMNS
                result['columns'] = True
            except:
                result['columns'] = False
            try:
                x['date'] = pd.to_datetime(x['date'], format = datetime_format)
                result['to_datetime'] = True
            except:
                result['to_datetime'] = False
            try:
                x = x.sort_values(by = 'date')
                result['sort_values'] = True
            except:
                result['sort_values'] = False
            try:
                x.set_index('date', inplace = True)
                result['set_index'] = True
            except:
                result['set_index'] = False
            result['read_scv'] = True
        except:
            result['read_scv'] = False
        return result, x

    def local(self, name, path):
        "Качаем дату с локальной папки (по умолчанию)."
        parse_from = (path + self.key + '/' + name + '.csv')
        datetime_format = '%Y.%m.%d'
        result, x = self.parse_csv(name, parse_from, datetime_format)
        return result, x

    def web(self, name, path):
        "Качаем дату с сайта."
        parse_from = (self.PATH_WEB_1 + name)
        datetime_format = '%d.%m.%Y'
        result, x = self.parse_csv(name, parse_from, datetime_format)
        if result['read_scv']:
            try:
                x.to_csv(path + self.key + '/' + name + '.csv')
            except:
                print(name, 'no save')
        return result, x

    def parse(self, name):
        """
        argument name as follows: RTS-3.23, RTS-3.23M150922CA100000 (no .csv extension)
        """
        if len(name) > 10:
            path = self.PATH_LOCAL_OPTIONS
        else:
            path = self.PATH_LOCAL_FUTURES
        result, x = self.local(name, path)
        result['parse_local'] = True
        if result['read_scv'] == False:
            result, x = self.web(name, path)
            result['parse_local'] = False
        try:
            x = x[[self.TARGET_COLUMN]]
            result['set_t_col'] = True
        except:
            result['set_t_col'] = False
        return result, x
    
    #-----качаем фьючерс
    def fut(self):
        name = self.name
        result, x = self.parse(name)
        result = pd.DataFrame([result])
        return result, x

    #-----добавляем опционы на фьючерс        
    def opt(self):
        results = []
        options_df = pd.DataFrame()
        for strike in range(self.lower - self.N_STRIKES * self.strike_step, self.upper + (self.N_STRIKES + 1) \
                            * self.strike_step, self.strike_step):
            for call_put in self.CALL_PUT:
                name = self.name + 'M' + self.exp_date + call_put + 'A' + str(strike)
                result, data = self.parse(name)
                if result['read_scv'] == False:
                    name = self.name + 'M' + self.exp_date + call_put + 'A' + '%20' + str(strike)
                    result, data = self.parse(name)
                try:
                    data['strike'] = strike
                    result['strike'] = True
                except:
                    result['strike'] = False
                try:
                    data['option_type'] = self.CALL_PUT[call_put]
                    result['option_type'] = True
                except:
                    result['option_type'] = False
                results.append(result)
                options_df = pd.concat([options_df, data])
        try:
            options_df = options_df.sort_values(by = ['date', 'strike', 'option_type'])
        except:
            print('no sort opt df')
        results = pd.DataFrame(results)
        results = results.fillna(0)
        return results, options_df
        
    #-----ниже накладываем фичи на датафрейм фьючерса
    def calc_rv(self, shift = 0):
        """
        Накладываем фичи реализованной волатильности.
        В целях предсказания делаем сдвиг фичей:
        """
        self.futures_df['return'] = self.futures_df[self.TARGET_COLUMN].shift(shift).pct_change() + 1
        self.futures_df['ln_r'] = np.log(self.futures_df['return'])
        self.futures_df['volatility_5'] = self.futures_df['ln_r'].rolling(5).std(ddof = 1) * 100 * np.sqrt(260)
        self.futures_df['volatility_20'] = self.futures_df['ln_r'].rolling(20).std(ddof = 1) * 100 * np.sqrt(260)
        self.futures_df['volatility_60'] = self.futures_df['ln_r'].rolling(60).std(ddof = 1) * 100 * np.sqrt(260)

    def movings(self, n = 5, shift = 0):
        """
        Накладываем мувинги.
        Параметры:
        n - период усреднения, обязательный параметр, по умолчанию равен пяти.
        shift - сдвиг назад: в целях расечта предсказательных моделей необходимо добовалять в целях
                предотвращения утечки целевого признака, в обзорных целях нагляднее не применять сдвиг.
                по умолчанию сдвиг равен единице.
        """
        self.futures_df['mov_{}'.format(n)] = self.futures_df[self.TARGET_COLUMN].shift(shift).rolling(n).mean()
            
    def lags(self):
        """
        Эта функция накладывает лаги цены.
        """
        self.futures_df['lag_1'] = self.futures_df[self.TARGET_COLUMN].shift(1)
        self.futures_df['lag_2'] = self.futures_df[self.TARGET_COLUMN].shift(2)
        self.futures_df['lag_3'] = self.futures_df[self.TARGET_COLUMN].shift(3)
        self.futures_df['lag_4'] = self.futures_df[self.TARGET_COLUMN].shift(4)
        self.futures_df['lag_5'] = self.futures_df[self.TARGET_COLUMN].shift(5)
        self.futures_df['lag_6'] = self.futures_df[self.TARGET_COLUMN].shift(6)
        self.futures_df['lag_7'] = self.futures_df[self.TARGET_COLUMN].shift(7)
        self.futures_df['lag_8'] = self.futures_df[self.TARGET_COLUMN].shift(8)
        self.futures_df['lag_9'] = self.futures_df[self.TARGET_COLUMN].shift(9)
        self.futures_df['lag_10'] = self.futures_df[self.TARGET_COLUMN].shift(10)
        self.futures_df['lag_20'] = self.futures_df[self.TARGET_COLUMN].shift(20)
        self.futures_df['lag_40'] = self.futures_df[self.TARGET_COLUMN].shift(40)
        self.futures_df['lag_60'] = self.futures_df[self.TARGET_COLUMN].shift(60)

    def diffs(self):
        """
        Эта функция накладывает разницу цены.
        """
        self.futures_df['dif_1'] = self.futures_df[self.TARGET_COLUMN] - self.futures_df[self.TARGET_COLUMN].shift(1)
        self.futures_df['dif_2'] = self.futures_df[self.TARGET_COLUMN] - self.futures_df[self.TARGET_COLUMN].shift(2)
        self.futures_df['dif_3'] = self.futures_df[self.TARGET_COLUMN] - self.futures_df[self.TARGET_COLUMN].shift(3)
        self.futures_df['dif_4'] = self.futures_df[self.TARGET_COLUMN] - self.futures_df[self.TARGET_COLUMN].shift(4)
        self.futures_df['dif_5'] = self.futures_df[self.TARGET_COLUMN] - self.futures_df[self.TARGET_COLUMN].shift(5)
        self.futures_df['dif_10'] = self.futures_df[self.TARGET_COLUMN] - self.futures_df[self.TARGET_COLUMN].shift(10)
        self.futures_df['dif_20'] = self.futures_df[self.TARGET_COLUMN] - self.futures_df[self.TARGET_COLUMN].shift(20)
        self.futures_df['dif_40'] = self.futures_df[self.TARGET_COLUMN] - self.futures_df[self.TARGET_COLUMN].shift(40)
        self.futures_df['dif_60'] = self.futures_df[self.TARGET_COLUMN] - self.futures_df[self.TARGET_COLUMN].shift(60)

    def calendar_features(self):
        """
        Эта функция накладывает календарные фичи.
        """
        # константа и тренд
        self.futures_df['const'] = np.ones(self.futures_df.shape[0])
        self.futures_df['time_step'] = np.arange(len(self.futures_df.index))
        # сезонность
        self.futures_df['year'] = self.futures_df.index.year
        self.futures_df['month'] = self.futures_df.index.month
        self.futures_df['day'] = self.futures_df.index.day
        self.futures_df['day_of_week'] = self.futures_df.index.dayofweek
        self.futures_df['sin_month'] = np.sin(2 * np.pi * self.futures_df.index.month / 12)
        self.futures_df['cos_month'] = np.cos(2 * np.pi * self.futures_df.index.month / 12)
        self.futures_df['sin_day'] = np.sin(2 * np.pi * self.futures_df.index.day / (365/12))
        self.futures_df['cos_day'] = np.cos(2 * np.pi * self.futures_df.index.day / (365/12))
        self.futures_df['sin_dw'] = np.sin(2 * np.pi * self.futures_df.index.dayofweek / 4)
        self.futures_df['cos_dw'] = np.cos(2 * np.pi * self.futures_df.index.dayofweek / 4)

    #-----сводное инфо по классу
    def info(self):
        try:
            print('Name:', self.name)
            print()
        except:
            pass
        print('Fut info:')
        try:
            print('Parse_log:')
            display(self.parse_fut_log)
            print()
        except:
            pass
        print('Head:')
        display(self.futures_df.head())
        print()
        print('Tail:')
        display(self.futures_df.tail())
        print()
        print('Info:')
        print(self.futures_df.info())
        print()
        print('Missings:')
        print(self.futures_df.isna().sum().sort_values(ascending = False))
        print()
        print('Monotonic:')
        print(self.futures_df.index.is_monotonic)
        print()
        print('Describe:')
        display(self.futures_df.describe())
        
        print('Opt info:')
        try:
            print('Parse_log:')
            display(self.qrt_opt_parse_log)
            print()
        except:
            pass
        print('Opt parse log:')
        print(self.qrt_opt_parse_log.mean())
        print('Head:')
        display(self.qrt_opt.head())
        print()
        print('Tail:')
        display(self.qrt_opt.tail())
        print()
        print('Info:')
        print(self.qrt_opt.info())
        print()
        print('Missings:')
        print(self.qrt_opt.isna().sum().sort_values(ascending = False))
        print()
        print('Monotonic:')
        print(self.qrt_opt.index.is_monotonic)
        print()
        print('Describe:')
        display(self.qrt_opt.describe())
        print('-----')
        
class Vision:
    def __init__(self, name_):
        self.TARGET = TARGET_COLUMN_NAME
        self.MONTHS = ['3', '6', '9', '12']
        self.YEARS = ['17', '18', '19', '20', '21']
        self.names_list = []
        self.objects_dict = {}
        self.name = name_
        #-----let's fill dict with J. objects
        for year in self.YEARS:
            for month in self.MONTHS:
                name = self.name + '-' + str(month) + '.' + str(year)
                self.names_list.append(name)
                self.objects_dict[name] = Jarvis(name)
                
    #-----we need some methods for most common analysis patterns
    def plot(self, what_to_plot = 'fut', value = 'adjp'):
        start_date = '2000-01-01'
        if what_to_plot == 'fut':
            targets = pd.DataFrame()
            for name in self.names_list:
                object_ = self.objects_dict[name]
                target = object_.futures_df[[value]]
                target = target[start_date:]
                start_date = str(target.index[target.shape[0]-1].year) + '-' + \
                str(target.index[target.shape[0]-1].month) + '-' + \
                str(target.index[target.shape[0]-1].day+1)
                targets = pd.concat([targets, target])
            targets[self.TARGET].plot(figsize = (15, 9))

class Tony:
    def __init__(self):
        self.KEYS = ['RTS', 'Si', 'GAZR', 'SBRF']