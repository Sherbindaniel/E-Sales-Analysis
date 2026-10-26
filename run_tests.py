import unittest
import pandas as pd
import numpy as np
import os
import sys

# Ensure src/ is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data_loader import load_raw_ecommerce_data, load_raw_web_server_data
from preprocessing import clean_ecommerce_data, clean_web_server_data
from feature_engineering import aggregate_ecommerce_to_minutes, aggregate_web_server_to_minutes
from distributions import check_overdispersion, fit_poisson, fit_nbinom, chi_square_gof
from anomaly_detection import detect_zscore_anomalies, detect_iqr_anomalies

class TestDataSciencePipeline(unittest.TestCase):
    
    def setUp(self):
        # Create a small dummy dataframe representing E-commerce raw log
        self.dummy_ec = pd.DataFrame({
            'accessed_date': [
                '2017-03-14 17:43:00.000',
                '2017-03-14 17:43:30.000',
                '2017-03-14 17:45:00.000' # note: 17:44 is missing!
            ],
            'ip': ['1.1.1.1', '2.2.2.2', '3.3.3.3'],
            'sales': [100.0, 50.0, 20.0],
            'duration_(secs)': [100, 200, 300],
            'bytes': [1000, 2000, 3000],
            'age': ['25', '--', '30'] # contains '--' string
        })
        
        # Create a small dummy access log
        self.dummy_access = pd.DataFrame({
            'timestamp': ['2024-01-01 00:00:00', '2024-01-01 00:00:30', '2024-01-01 00:02:00'],
            'server_id': [1, 2, 1],
            'method': ['GET', 'POST', 'GET'],
            'path': ['/', '/api', '/'],
            'status_code': [200, 404, 200],
            'response_time_ms': [100, 200, 150],
            'bytes_sent': [1000, 500, 1200],
            'user_agent': ['curl', 'chrome', 'firefox'],
            'ip_address': ['1.1.1.1', '2.2.2.2', '1.1.1.1'],
            'referrer': ['-', '-', '-']
        })
        
        self.dummy_servers = pd.DataFrame({
            'id': [1, 2],
            'name': ['web-1', 'web-2'],
            'region': ['us-east', 'us-east'],
            'role': ['api', 'api']
        })

    def test_preprocessing_ecommerce_age_and_dates(self):
        """
        Verify preprocessing E-commerce parses datetime and maps age '--' to NaN.
        """
        df_clean = clean_ecommerce_data(self.dummy_ec)
        
        # Verify age '--' is replaced with NaN
        self.assertTrue(pd.isna(df_clean['age'].iloc[1]))
        self.assertEqual(df_clean['age'].iloc[0], 25.0)
        self.assertEqual(df_clean['age'].iloc[2], 30.0)
        
        # Verify datetime conversion
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_clean['accessed_date']))

    def test_preprocessing_web_server_join(self):
        """
        Verify preprocessing web server access logs joins server metadata and parses datetime.
        """
        df_joined = clean_web_server_data(self.dummy_access, self.dummy_servers)
        
        # Check that server metadata columns are joined
        self.assertIn('name', df_joined.columns)
        self.assertIn('region', df_joined.columns)
        self.assertEqual(len(df_joined), 3)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_joined['timestamp']))

    def test_feature_engineering_reindexing(self):
        """
        Verify that 1-minute aggregation fills missing minutes with zero count.
        """
        df_clean = clean_ecommerce_data(self.dummy_ec)
        agg_df, p90 = aggregate_ecommerce_to_minutes(df_clean)
        
        # Max is 17:45, Min is 17:43. Total minutes = 3 (17:43, 17:44, 17:45)
        self.assertEqual(len(agg_df), 3)
        
        # Verify that minute 17:44 has requests_per_minute = 0
        df_1744 = agg_df[agg_df['minute_bin'] == '2017-03-14 17:44:00']
        self.assertEqual(df_1744['requests_per_minute'].iloc[0], 0)
        
        # Verify that minute 17:43 has requests_per_minute = 2
        df_1743 = agg_df[agg_df['minute_bin'] == '2017-03-14 17:43:00']
        self.assertEqual(df_1743['requests_per_minute'].iloc[0], 2)

    def test_distributions_overdispersion_checks(self):
        """
        Test dispersion index calculations.
        """
        data = np.array([2, 2, 2, 2, 2]) # variance is 0
        mean, var, disp, is_od = check_overdispersion(data)
        self.assertEqual(var, 0)
        self.assertEqual(disp, 0)
        self.assertFalse(is_od)
        
        data_od = np.array([1, 1, 1, 1, 100]) # highly overdispersed
        mean_od, var_od, disp_od, is_od = check_overdispersion(data_od)
        self.assertTrue(disp_od > 1.2)
        self.assertTrue(is_od)

    def test_distributions_fitting(self):
        """
        Verify MLE fits and Chi-Square GoF tests run without errors.
        """
        # Poisson fit
        data = np.array([1, 2, 1, 3, 2, 1, 1, 2, 0, 1, 12, 1, 2, 1, 0, 1])
        poi_params = fit_poisson(data)
        self.assertAlmostEqual(poi_params[0], np.mean(data))
        
        # Negative Binomial fit
        nb_params = fit_nbinom(data)
        self.assertEqual(len(nb_params), 2)
        self.assertTrue(0 < nb_params[1] < 1) # p-value must be between 0 and 1
        
        # Chi-Square test
        # We need a larger dataset so that bins can have expected counts >= 5
        large_data = np.random.poisson(lam=5, size=200)
        large_poi_params = fit_poisson(large_data)
        chi, p, dof = chi_square_gof(large_data, 'poisson', large_poi_params)
        self.assertFalse(np.isnan(chi))
        self.assertTrue(dof > 0)

    def test_anomaly_detection(self):
        """
        Verify Z-score and IQR anomaly checks return correct boolean masks.
        """
        series = pd.Series([1.0, 1.1, 1.2, 0.9, 1.0, 100.0, 1.1])
        z_anom = detect_zscore_anomalies(series, threshold=2.0)
        iqr_anom = detect_iqr_anomalies(series)
        
        self.assertTrue(z_anom.iloc[5]) # 100.0 is an anomaly
        self.assertTrue(iqr_anom.iloc[5])
        self.assertFalse(z_anom.iloc[0]) # 1.0 is not an anomaly

if __name__ == '__main__':
    unittest.main()
