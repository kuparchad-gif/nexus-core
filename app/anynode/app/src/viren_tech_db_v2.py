from kaggle.api.kaggle_api_extended import KaggleApi
from tinydb_farm import TinyDBFarm
from vault_carrier import VaultCarrier
import os
import uuid

class VirenTechDatabase:
    def __init__(self):
        self.vault = VaultCarrier()
        self.api = KaggleApi()
        self.api.authenticate()  # Uses ~/.kaggle/kaggle.json
        self.farms = [TinyDBFarm(f'farm_{i}') for i in range(10)]

    def update_tech_data(self):
        data = {'timestamp': str(uuid.uuid4()), 'tech': 'VIREN', 'status': 'active'}
        farm = self.farms[hash(data['timestamp']) % len(self.farms)]
        farm.write_record('tech_data', data)

    def scrape_kaggle_datasets(self, query, max_datasets=5):
        datasets = self.api.dataset_list(search=query, license='cc0-public-domain')
        for dataset in datasets[:max_datasets]:
            self.api.dataset_download_files(dataset.ref, path=f'/app/kaggle/{dataset.ref}', unzip=True)
            for file in os.listdir(f'/app/kaggle/{dataset.ref}'):
                with open(f'/app/kaggle/{dataset.ref}/{file}', 'r') as f:
                    data = f.read()
                    farm = self.farms[hash(file) % len(self.farms)]
                    farm.write_record('kaggle_data', {'file': file, 'data': data})