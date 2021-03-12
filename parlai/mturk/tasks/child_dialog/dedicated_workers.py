import json
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
import gspread


class ReviewGSheet:
    """
    Class to convert the Google Sheets into Pandas.DataFrame
    """

    def __init__(self, credential_file):
        """
        @g_auth: google sheet authorization object
        """
        self.scope = ['https://www.googleapis.com/auth/spreadsheets.readonly']

        credentials = ServiceAccountCredentials.from_json_keyfile_name(credential_file, self.scope)
        self.client = gspread.authorize(credentials)
        with open(credential_file, 'r') as cred_f:
            cred = json.load(cred_f)
        self.spreadsheet_link = cred['spreadsheet_link']
        self.sheet_num = cred['sheet_num']

    def convert_gsheets_to_dataframe(self, link: str = None, sheets_num: int = None) -> pd.DataFrame:
        """
        @link: link of the gsheets. Note: google sheets need to be public to access from google.auth
        @sheets_num: Sheets number of google sheets class
        return: DataFrame of google sheets
        """
        if link is None:
            link = self.spreadsheet_link
        if sheets_num is None:
            sheets_num = self.sheet_num
        data = self.client.open_by_url(link).get_worksheet(sheets_num).get_all_values()
        return pd.DataFrame.from_records(data[1:], columns=data[0])

    def upload_sheet(self, link: str, sheets_num: int, csv_path):
        """
        @link: link of the gsheets. Note: google sheets need to be public to access from google.auth
        @sheets_num: Sheets number of google sheets class
        """
        sheet_id = self.client.open_by_url(link).id
        print(f'sheet id to which the csv file will uploaded is : {sheet_id}')
        with open(csv_path, 'r') as content:
            self.client.import_csv(sheet_id, content.read())

    def get_review_data(self, link: str = None, sheets_num: int = None):
        return self.convert_gsheets_to_dataframe(link, sheets_num)

    def get_golden_workers_list(self, link: str = None, sheets_num: int = None, exclude_workers=[]):
        review_df = self.get_review_data(link, sheets_num)
        worker_list = review_df[(review_df['Points'].isin(['Golden', 'Golden 20']))
                                & (review_df['DATE'].isin(['8/25/2020', '8/26/2020', '8/27/2020']))
                                & ~(review_df['Worker ID'].isin(exclude_workers))][
            'Worker ID'].dropna().unique().tolist()
        # all_golden_50 = review_df[(review_df['Points'] == 'Golden 50') & (review_df['DATE'] == '8/24/2020')]
        # worker_golden50_count = all_golden_50.groupby('Worker ID')['Points'].agg('count')
        # return list(worker_golden50_count[worker_golden50_count > 2].index)
        return worker_list