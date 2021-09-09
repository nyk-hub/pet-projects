import sys
import pandas as pd
import os
import pickle as pkl
import numpy as np
import catboost


class Bankrupcy:
    def __init__(self):
        self.SEED = 42
        self.curr_dir = os.getcwd()
        self.data_dir = os.path.join(self.curr_dir, 'data')
        self.cases_dir = os.path.join(self.curr_dir,
                                      'data/court_cases_sample/success')
        self.model = self.get_model()

    def get_model(self):
        if os.path.exists(os.path.join(self.curr_dir, 'model.pkl')):
            print('Загружаем предобученную модель')
            with open(os.path.join(
                    self.curr_dir, 'model.pkl'), 'rb') as f:
                model = pkl.load(f)
            print('done')
        else:
            print('Обучаем модель')
            params = {'learning_rate': 0.1,
                      'depth': 9,
                      'n_estimators': 40,
                      'l2_leaf_reg': 6
                      }
            cat_cols = ['year', 'inn']

            model = catboost.CatBoostClassifier(cat_features=cat_cols,
                                                eval_metric='AUC',
                                                random_seed=self.SEED)
            model.set_params(**params)
            data = self.preprocess()
            X_train = data.drop('bankrupt', axis=1)
            y_train = data.bankrupt

            model.fit(X_train, y_train, silent=True)

            with open(os.path.join(
                    self.curr_dir, 'model.pkl'), 'wb') as f:
                pkl.dump(model, f)
            print('done')

        return model

    def predict(self, pkl_file, csv1_file, csv2_file):
        val_data = self.preprocess(pkl_file, csv1_file, csv2_file)
        self.inference(val_data)

    def inference(self, data):
        model = self.model
        data.drop('bankrupt', axis=1, inplace=True)
        preds = pd.Series(model.predict_proba(data)[:, 1])

        preds.to_csv(os.path.join(self.curr_dir, 'result.csv'),
                     header=['probability'], index=False
                     )
        print('Предсказания сохранены в файл: result.csv')
        return preds

    def preprocess(self,
                   cases=None,
                   accounts='df_accounts_sample.csv',
                   bankruptcies='df_bankruptcies_sample.csv'
                   ):
        def parse_pkl(path):
            def parse_file(file, columns):
                curr_frame = pd.DataFrame(columns=columns)
                with open(file, 'rb') as f:
                    dict = pkl.load(f)

                inn = dict['inn']
                for case in dict['cases_list']:
                    year = case['caseDate'][:4]
                    result = case['resultType']
                    money = case['sum']
                    currency = case['currency']
                    for side in case['case_sides']:
                        if str(inn) in side['INN']:
                            case_type = side['type']

                    line = pd.DataFrame([[inn, year, result,
                                          money, currency,
                                          case_type]],
                                        columns=columns)
                    curr_frame = pd.concat([curr_frame, line],
                                           axis=0)

                return curr_frame

            columns = ['inn', 'year', 'result', 'money', 'currency',
                       'case_type']

            df_cases = pd.DataFrame(columns=columns)
            if not path:
                print('Получаем информацию из .pkl файлов')
                for file in os.listdir(self.cases_dir):
                    if file.endswith('.pkl'):
                        df_cases = pd.concat(
                            [df_cases, parse_file(
                                os.path.join(self.cases_dir, file),
                                columns)], axis=0)
                print('done')
            else:
                df_cases = parse_file(
                    os.path.join(self.data_dir, path), columns)

            return df_cases

        def okei(line):
            mult = {384: 1e3, 385: 1e6}
            if line.okei in mult:
                coef = mult[line.okei]
                line.long_term_liabilities_fiscal_year *= coef
                line.short_term_liabilities_fiscal_year *= coef
                line.balance_assets_fiscal_year *= coef

            return line

        def result_process(line):
            unkn = {'Не удалось определить', 'Иск не рассмотрен',
                    'Прекращено производство по делу',
                    'Утверждено мировое соглашение'}
            loose = {'Проиграно', 'Частично проиграно', 'Не выиграно',
                     'В иске отказано полностью', }
            win = {
                'Выиграно', 'Частично выиграно', 'Не проиграно',
                'Иск полностью удовлетворен',
                'Иск частично удовлетворен',
                'В иске отказано частично',
                'Иск полностью удовлетворен, встречный частично удовлетворен',
                'Иск частично удовлетворен, встречный не удовлетворен'
            }
            if line in unkn:
                return -1
            if line in loose:
                return 0
            if line in win:
                return 1

            return 0

        accounts_sample = pd.read_csv(
            os.path.join(self.data_dir, accounts), index_col=0)

        bankruptcies_sample = pd.read_csv(
            os.path.join(self.data_dir, bankruptcies), index_col=0)

        if not cases:
            if os.path.exists(os.path.join(self.data_dir, 'cases.pkl')):
                with open(os.path.join(self.data_dir, 'cases.pkl'),
                          'rb') as f:
                    df_cases = pkl.load(f)
            else:
                df_cases = parse_pkl(cases)
                with open(os.path.join(
                        self.data_dir, 'cases.pkl'), 'wb') as f:
                    pkl.dump(df_cases, f)
        else:
            df_cases = parse_pkl(cases)

        df_cases.year = df_cases.year.astype('int')
        df_cases.inn = df_cases.inn.astype('int64')
        df_cases.drop('currency', axis=1, inplace=True)
        df_cases.result = df_cases.result.apply(result_process)
        df_cases.case_type = df_cases.case_type.astype('int')

        accounts_sample = accounts_sample.apply(okei, axis=1)
        accounts_sample.drop('okei', axis=1, inplace=True)
        bankruptcies_sample.drop('bankrupt_id', axis=1, inplace=True)
        accounts_sample.year = accounts_sample.year.astype('int')
        accounts_sample.inn = accounts_sample.inn.astype('int64')

        for year, inn in bankruptcies_sample.values:
            idx = accounts_sample[(accounts_sample.inn == inn) & (
                    accounts_sample.year == year)].index
            accounts_sample.loc[idx, 'bankrupt'] = 1

        accounts_sample.bankrupt.fillna(0, inplace=True)
        accounts_sample.long_term_liabilities_fiscal_year.fillna(
            0, inplace=True)
        new_cols = ['case_0_count', 'case_1_count',
                    'case_0_result_1_count', 'case_0_result_0_count',
                    'case_1_result_0_count', 'case_1_result_1_count'
                    ]
        tmp_frame = pd.DataFrame(columns=new_cols)
        accounts_sample = pd.concat(
            [accounts_sample, tmp_frame], axis=1)

        tmp = df_cases.groupby(['inn', 'year'])[
            'case_type'].value_counts()
        for (inn, year, case_type), count in zip(tmp.index, tmp.values):
            if case_type in (0, 1):
                idx = accounts_sample[(accounts_sample.inn == inn) & (
                        accounts_sample.year == year)].index
                if idx is not None:
                    accounts_sample.loc[idx, 'case_' + str(
                        case_type) + '_count'] = count

        tmp = df_cases.groupby(['inn', 'year', 'case_type'])[
            'result'].value_counts()
        for (inn, year, case_type, case_result), count in zip(
                tmp.index, tmp.values):
            if case_type in (0, 1) and case_result in (0, 1):
                idx = accounts_sample[(accounts_sample.inn == inn) & (
                        accounts_sample.year == year)].index
                if idx is not None:
                    accounts_sample.loc[idx, 'case_' + str(
                        case_type) + '_result_' + str(
                        case_result) + '_count'] = count

        accounts_sample.fillna(0, inplace=True)
        accounts_sample.year = accounts_sample.year.astype('str')
        accounts_sample.inn = accounts_sample.inn.astype('str')

        return accounts_sample


def main():
    print('Подайте файлы в следующей очередности: \n \
           судебный процессов .pkl \n \
           accounts_sample .csv \n \
           bankruptcies_sample .csv')

    cases = sys.stdin.readline().strip()
    accounts = sys.stdin.readline().strip()
    bankrupcy_df = sys.stdin.readline().strip()

    b = Bankrupcy()
    b.predict(cases, accounts, bankrupcy_df)


if __name__ == "__main__":
    main()
